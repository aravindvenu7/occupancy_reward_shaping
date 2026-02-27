import collections
import os
import platform
import time
import random
import gymnasium
import numpy as np
from gymnasium.spaces import Box
import tqdm
import ogbench
from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name, append_fp_states = False, frame_stack=None, abstract = 1, discount = 0.99, encoder_only = False):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    # Use compact dataset to save memory.
    print("Creating compact dataset.")
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)
    if append_fp_states == 1:
        dones = 1 - train_dataset["valids"]
        next_dones, prev_dones = get_next_dones(dones), get_prev_dones(dones)        
        future_obs_idxes = sample_fp_states(next_dones, prev_dones, abstract = abstract, discount = discount)
        train_dataset['future_obs_idx'] = future_obs_idxes
        
    train_dataset = Dataset.create(**train_dataset)
    
    if append_fp_states == 1:
        dones = 1 - val_dataset["valids"]
        next_dones, prev_dones = get_next_dones(dones), get_prev_dones(dones)        
        future_obs_idxes = sample_fp_states(next_dones, prev_dones, abstract = abstract, discount = discount)
        val_dataset['future_obs_idx'] = future_obs_idxes
        
    val_dataset = Dataset.create(**val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()

    return env, train_dataset, val_dataset


def get_next_dones(terminals):
    
    next_terminals = np.zeros(terminals.shape[0])
    future_t = terminals.shape[0] - 1
    for i in range(terminals.shape[0] - 1, -1, -1):
      if bool(terminals[i]):
        future_t = i
      next_terminals[i] = future_t
    next_terminals = np.array(next_terminals)
    return next_terminals


def get_prev_dones(terminals):
    
    prev_terminals = np.zeros(terminals.shape[0])
    past_t = 0
    for i in range(0, terminals.shape[0] - 1):
      prev_terminals[i] = past_t
      if bool(terminals[i]):
        past_t = i + 1
      
    prev_terminals = np.array(prev_terminals)
    return prev_terminals

def get_prev_obs_idx(terminals):
    prev_obs_idx = np.zeros(terminals.shape[0])
    for i in range(1, terminals.shape[0]):
        if bool(terminals[i - 1]):
            prev_obs_idx[i] = i
        else:          
            prev_obs_idx[i] = i - 1
    return prev_obs_idx.astype(int)

    
#given a state s_t, sample the future state distribution with probabilities starting from s_t+1
def sample_fp_states(next_terminal, prev_terminal, max_horizon = 999, n_samples = 50, abstract = 1, discount = 0.99):

      
      print("sampling future states...")
      gamma = discount
      obs_indices = np.arange(next_terminal.shape[0])
      horizon = next_terminal - (obs_indices)
      ret_arr = np.tile(np.arange((max_horizon)), (horizon.shape[0], 1))
      #a_ret_arr = np.tile(np.arange((max_horizon//abstract) + abstract), (horizon.shape[0], 1))
      a_ret_arr = np.tile(np.arange((max_horizon)), (horizon.shape[0], 1))
      probs = gamma**a_ret_arr
      probs = (1-gamma)*probs
      #probs = np.repeat(probs, abstract, axis = 1)[:,:max_horizon]
      mask = ret_arr <= (horizon[:, None]) #truncating the distribution may introduce some bias.
      probs *= mask
      probs /= np.sum(probs, axis=1, keepdims=1)
          
      delta = []
      for i in tqdm.tqdm(range(len(horizon))):
          assert horizon[i] >= 0
          chosen_idx = np.random.choice(ret_arr[i], n_samples, p=probs[i]) 
          # +1 since we're going to be selecting these idxes from observations array
          selected_idx = (chosen_idx + i + 1).astype(int)
          selected_idx[selected_idx >= len(next_terminal)] = len(next_terminal) - 1
          assert np.all(selected_idx < len(next_terminal)), print(selected_idx)
          delta.append(selected_idx)
          
      future_obs_idx = [item.astype(int) for item in delta]
      future_obs_idx = np.stack(future_obs_idx, axis = 0)

    
      return future_obs_idx

def get_ogbench_dataset_by_trajectory(dataset=None, terminate_on_end=False, **kwargs):

    N = dataset["actions"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    done_ = []


    episodes = []
    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        done_bool = bool(dataset["valids"][i])

        if not done_bool:
            # record this episode and reset the stats
            episode = {
                "observations": np.array(obs_),
                "actions": np.array(action_),
                "next_observations": np.array(next_obs_),
                "terminals": ~ np.array(done_),
            }
            episodes.append(episode)

            episode_step = 0
            obs_ = []
            next_obs_ = []
            action_ = []
            done_ = []


        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        done_.append(done_bool)
        episode_step += 1

    return episodes


def get_n_trajectories(dataset, n):
    traj_dataset = get_ogbench_dataset_by_trajectory(dataset)
    n_traj = []
    #n = random.sample(range(0, len(traj_dataset)-1), n)
    for num in range(n):
        n_traj.append(traj_dataset[num]['next_observations'])
    return n_traj

def get_optimal_trajectories(path):
    n_traj = []
    for i in range(5):
        traj = np.load(os.path.join(path, str(i+1)+".npy"), allow_pickle = True)
        traj = np.array(traj.item()['observations'])
        n_traj.append(traj)
    return n_traj 