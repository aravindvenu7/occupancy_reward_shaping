from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
import jax.numpy as jnp

def smooth_array(arr, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='same')


def compute_shaped_rewards(forward_model, inits, ends, num_samples = 200):

    batch_size, observation_dim = inits.shape[0], inits.shape[-1]
    ends = jnp.reshape(jnp.repeat(jnp.expand_dims(ends, axis=-2), num_samples, axis=-2), (-1, ends.shape[-1]))
    inits = jnp.reshape(jnp.repeat(jnp.expand_dims(inits, axis=-2), num_samples, axis=-2), (-1, inits.shape[-1]))

    x_rng = forward_model.rng
    x_0 = jax.random.normal(x_rng, (batch_size * num_samples, observation_dim))
    x_1 = ends 
    vel = x_1 - x_0
    flow_steps = int(forward_model.config['flow_steps']*3.5)#//2
    rewards = []
    for i in range(3, flow_steps, 3):
        t = jnp.full((*inits.shape[:-1], 1), i / flow_steps)
        x_t = (1 - t) * x_0 + t * x_1
        pred_f = forward_model.network.select('actor_bc_flow')(inits, x_t, t)
        pred_b = vel
        reward = (pred_f - pred_b) ** 2
        reward = jnp.reshape(reward, (batch_size, num_samples, observation_dim))
        reward = jnp.mean(reward, axis=(1, 2))
        rewards.append(reward)
        
    rewards = jnp.stack(rewards, axis=1)
    rewards = jnp.mean(rewards, axis=1)
    assert rewards.shape[0] ==  batch_size, "Size mismatch in reward computation."
    rewards = jax.lax.stop_gradient(rewards)
    return rewards

def compute_l2_rewards(forward_model, inits, ends, num_samples = 200):

    rewards = np.sqrt((ends - inits)**2).mean(axis = -1)
    return smooth_array(rewards, window_size = 1)
    
def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def compute_nmc_rate(values):

    nmc = values[1:] < values[:-1]
    nmc_rate = np.sum(nmc)/len(nmc)
    return nmc_rate

def discounted_sum(arr, gamma=0.99):

    discounts = np.arange(len(arr))**gamma
    arr = arr*discounts
    return arr.sum()
    
def discounted_rewards(rewards, gamma=0.99, abstract = 1):
    if abstract == 1:
        discounted = np.zeros_like(rewards, dtype=np.float32)
        future_cumulative = 0
        
        #np.random.choice([1, -1], size=values.shape[0]).astype(float) * 0.01 * values
        # Iterate backwards through time steps
        for t in reversed(range(len(rewards))):
            noise = np.random.normal(loc=0.0, scale=0.0005)
            future_cumulative = rewards[t] + gamma * (future_cumulative + noise*future_cumulative)
            discounted[t] = future_cumulative
    else:
        discounted = np.zeros(len(rewards) - abstract)
        future_cumulative = 0

        for t in reversed(range(len(rewards) - abstract)):
            future_cumulative = np.sum(rewards[t:t+abstract]) + (gamma) * future_cumulative 
            #future_cumulative = (0.5*(rewards[t] + rewards[t+abstract]))+ (gamma) * future_cumulative 
            discounted[t] = future_cumulative
    
    return discounted
    
def evaluate_q_rs(
    agent,
    fwd_model,
    freq,
    env,
    trajectories,
    config=None,
    eval_gaussian=None,
    rew_model=None
):
    nmc_rate = []
    all_rewards = []
    all_values = []
    for i in trange(len(trajectories)):

        trajectory = trajectories[i][::freq]
        goal = np.tile(trajectory[-1], (trajectory.shape[0], 1))
        #rewards = -1 * compute_l2_rewards(fwd_model, trajectory, goal, num_samples = 400) #1000,1
        rewards = np.ones(trajectory.shape[0])*-1
        rewards[-1] = 0.0
        #for _ in range(1):
        if rew_model is None:
            rewards = -1.0 * compute_shaped_rewards(fwd_model, trajectory, goal, num_samples = 100) #1000,1
        else:
            rewards = rew_model.compute_reward(trajectory, goal)


        #all_rewards.append(rewards[:-1])
        values = discounted_rewards(rewards[:-1], abstract = 1)
        all_values.append(values)
        nmc_rate.append(compute_nmc_rate(values))

    return np.mean(nmc_rate), all_values



    
def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
