import json
import os
import random
import time
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets, sample_fp_states
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('project', 'OGBench', 'Wandb project')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 2000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 500000, 'Saving interval.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 2, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')
config_flags.DEFINE_config_file('agent', 'agents/hiql.py', lock_config=False)

#arguments related to reward shaping:
flags.DEFINE_integer('reward_shaping', 0, 'Whether to sample future states and append to dataset')
flags.DEFINE_integer('state_action', 0, 'Whether to use SA model to learn r(s,a)')
flags.DEFINE_integer('append_fp_states', 0, 'Whether to sample future states and append to dataset')
flags.DEFINE_string('future_restore_path', None, 'Restore path for future agent.')
flags.DEFINE_integer('fp_restore_epoch', None, 'Restore epoch for future agent.')
config_flags.DEFINE_config_file('rs_agent', 'agents/mc_occ.py', lock_config=False)

def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project=FLAGS.project, group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.env_name, exp_name)
    if bool(FLAGS.state_action):
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, "sa")
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    
    def safe_json(obj):
        if isinstance(obj, (str, int, float, bool, list, dict, type(None))):
            return obj
        else:
            return str(obj)  

    cleaned_dict = {k: safe_json(v) for k, v in flag_dict.items()}
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(cleaned_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    rs_config = FLAGS.rs_agent
    
    print("frame stack:", config['frame_stack'])
    
    config['reward_shaping'] = rs_config['reward_shaping'] = bool(FLAGS.reward_shaping)
    config['append_fp_states'] = rs_config['append_fp_states'] = bool(FLAGS.append_fp_states)
    config['state_action'] = rs_config['state_action'] = FLAGS.state_action
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    
    #add extra function to get the future states
    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)


    #create agents.
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    if FLAGS.reward_shaping:
            config['batch_size'] = FLAGS.batch_size
            agent_class = agents[rs_config['agent_name']]
            rs_config['future_prediction'] = True
            future_agent = agent_class.create(
                FLAGS.seed,
                example_batch['observations'],
                example_batch['actions'],
                rs_config,
            )


        
    # Restore agents.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    if FLAGS.reward_shaping:
        future_agent = restore_agent(future_agent, FLAGS.future_restore_path, FLAGS.fp_restore_epoch) 

        
    print("Agents restored. Ready to train.")
    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        if FLAGS.reward_shaping:
            batch = train_dataset.sample(config['batch_size'], fwd_model = future_agent)
        else:
            batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                if FLAGS.reward_shaping:
                    val_batch = val_dataset.sample(config['batch_size'], fwd_model = future_agent)
                else:
                    val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
