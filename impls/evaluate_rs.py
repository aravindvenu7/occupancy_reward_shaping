import json
import os
import random
import time
from collections import defaultdict
import ogbench
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets, get_ogbench_dataset_by_trajectory, get_n_trajectories, \
                            get_optimal_trajectories
from utils.evaluation import evaluate, evaluate_q_rs
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
import matplotlib.pyplot as plt
import seaborn as sns
FLAGS = flags.FLAGS


flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')
flags.DEFINE_bool('use_optimal', False, "use collected optimal trajectories for analysis")
config_flags.DEFINE_config_file('agent', 'agents/gcivl.py', lock_config=False)


#arguments related to reward shaping:
flags.DEFINE_integer('reward_shaping', 1, 'Whether to sample future states and append to dataset')
flags.DEFINE_integer('append_fp_states', 0, 'Whether to sample future states and append to dataset')
flags.DEFINE_string('future_restore_path', None, 'Restore path for future agent.')
flags.DEFINE_integer('fp_restore_epoch', None, 'Restore epoch for future agent.')
flags.DEFINE_integer('abstract', 1, 'Temporal abstraction for occupancy model')
config_flags.DEFINE_config_file('rs_agent', 'agents/mc_occ.py', lock_config=False)
config_flags.DEFINE_config_file('rew_agent', 'agents/rew.py', lock_config=False)
flags.DEFINE_string('rew_restore_path', None, 'Restore path for future agent.')
flags.DEFINE_string('rew_restore_epoch', None, 'Restore path for future agent.')
flags.DEFINE_bool('use_rew', False, "use reward model to infer reward")
flags.DEFINE_string('optimal_traj_path', None, 'Path for stored optimal trajectories.')

def main(_):
    # Set up logger.

    flag_dict = get_flag_dict()
    # Set up environment and dataset.
    config = FLAGS.agent
    occ_config = FLAGS.rs_agent
    rew_config = FLAGS.rew_agent
    config['reward_shaping'] = bool(FLAGS.reward_shaping)
    #config['use_rew'] = bool(FLAGS.use_rew)
    config['append_fp_states'] = bool(FLAGS.append_fp_states)
    
    #env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    if FLAGS.use_optimal:
        trajectories = get_optimal_trajectories(FLAGS.optimal_traj_path)
    else:
        trajectories = get_n_trajectories(train_dataset, 5)
    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    #Create agents
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    if FLAGS.reward_shaping:
            agent_class = agents[FLAGS.rs_agent['agent_name']]
            occ_config['future_prediction'] = True
            future_agent = agent_class.create(
                FLAGS.seed,
                example_batch['observations'],
                example_batch['actions'],
                occ_config,
            )
            if FLAGS.use_rew:
                agent_class = agents[FLAGS.rew_agent['agent_name']]
                rew_agent = agent_class.create(
                    FLAGS.seed,
                    example_batch['observations'],
                    example_batch['actions'],
                    rew_config,
                )        
    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

        
    if FLAGS.reward_shaping:
        if FLAGS.use_rew:
            rew_agent = restore_agent(rew_agent, FLAGS.rew_restore_path, FLAGS.rew_restore_epoch)
        else:
            future_agent = restore_agent(future_agent, FLAGS.future_restore_path, FLAGS.fp_restore_epoch)   
            rew_agent = None

    print("All agents restored. Ready to evaluate.")
    first_time = time.time()
    last_time = time.time()

    if FLAGS.eval_on_cpu:
        eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
    else:
        eval_agent = agent

    nmc_rates = []
    non_monotonicity_rate, all_rewards = evaluate_q_rs(
        agent=eval_agent,
        fwd_model=future_agent,
        freq = freq,
        env=env,
        trajectories = trajectories,
        config=config,
        eval_gaussian=FLAGS.eval_gaussian,
        rew_model = rew_agent
    )
    nmc_rates.append(non_monotonicity_rate)

    for i, rate in enumerate(nmc_rates):
        print(f"non-monotonicity rate: {rate}")

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, "reward_plots")
    os.makedirs(FLAGS.save_dir, exist_ok=True)


    sns.set(style="ticks")
    plt.figure(figsize=(8, 6))
    num_traj = len(all_rewards)
    for i in range(num_traj):
        plt.plot(all_rewards[i], label = str(i), linewidth=3)
    
    plt.grid(True, color="black", linewidth=1.2, linestyle="--", alpha=0.3, zorder=0)
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20) 
    plt.xlabel('t', fontsize = 25)
    plt.ylabel('$V_{(s,g)}$', fontsize=25)
    plt.title('Value, $\sigma_v$ = 0.0005', fontsize = 20)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(FLAGS.save_dir, 'rewards_rs.png'))  
    plt.close()

if __name__ == '__main__':
    app.run(main)
