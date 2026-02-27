import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
import random
import time
from collections import defaultdict

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

flags.DEFINE_string('run_group', 'occ_pretrain', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('future_restore_path', None, 'Restore path for future agent.')
flags.DEFINE_integer('future_restore_epoch', None, 'Restore epoch for future agent.')
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('abstract', 5, 'Temporal abstraction for occupancy model')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 250000, 'Saving interval.')
flags.DEFINE_float('discount', 0.99, 'Discount factor')

flags.DEFINE_integer('reward_shaping', 0, 'Whether to sample future states and append to dataset')
flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')
flags.DEFINE_integer('append_fp_states', 1, 'Whether to sample future states and append to dataset')
config_flags.DEFINE_config_file('agent', 'agents/mc_occ.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='occ_pretrain_large', group=FLAGS.run_group, name=exp_name)
    config = FLAGS.agent

        
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    
    future_save_dir = os.path.join(FLAGS.save_dir, "fa", str(FLAGS.abstract))
    if config['state_action']:
        future_save_dir = os.path.join(FLAGS.save_dir, "fa", "sa")
        
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    os.makedirs(future_save_dir, exist_ok=True)
    
    flag_dict = get_flag_dict()
    with open(os.path.join(future_save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    print("frame stack:", config['frame_stack'])
    
    config['reward_shaping'] = bool(FLAGS.reward_shaping)
    config['append_fp_states'] = bool(FLAGS.append_fp_states)
    config['discount'] = FLAGS.discount
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, FLAGS.append_fp_states, frame_stack=config['frame_stack'], abstract = FLAGS.abstract, discount = FLAGS.discount, encoder_only = config['encoder_only'])
    
    #add extra function to get the future states
    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    print("Datasets created.")
    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    future_agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    

    print("Agents created. Ready to train.")
    # Restore agent.
    if FLAGS.future_restore_path is not None:
        future_agent = restore_agent(future_agent, FLAGS.future_restore_path, FLAGS.future_restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    first_time = time.time()
    last_time = time.time()
    td_training = False
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):

        
        if i ==  int(FLAGS.train_steps//2):
            td_training = True
        # Update agent.
        
        batch = train_dataset.sample(config['batch_size'])
        future_agent, fa_update_info = future_agent.update(batch, td_training = td_training)

        # Log metrics.
        if i % FLAGS.log_interval != 0:
            train_metrics = {f'fa_training/{k}': v for k, v in fa_update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, fa_val_info = future_agent.total_loss(val_batch, grad_params=None)     
                train_metrics.update({f'fa_validation/{k}': v for k, v in fa_val_info.items()})
                
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time

            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(future_agent, future_save_dir, i)
    train_logger.close()


if __name__ == '__main__':
    app.run(main)
