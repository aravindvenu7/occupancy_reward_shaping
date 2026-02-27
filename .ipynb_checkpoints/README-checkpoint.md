# ORS on OGBench (Implementation)


This repo contains the official code for the ICLR 2026 paper:
"Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning"


The code trains ORS components and runs GCIQL with ORS reward shaping on OGBench environments.

## Requirements

This code uses the same dependencies as OGBench. Follow the OGBench setup instructions here:

```
https://github.com/seohongpark/ogbench/
```

## Quickstart

All commands below assume you run from `occupancy_reward_shaping/impls`.

### 1) Train the occupancy model

```bash
python occupancy.py \
  --env_name=antmaze-giant-navigate-v0 \
  --agent=agents/mc_occ.py \
  --train_steps 2000000 \
  --agent.state_action=True
```

### 2) Train the reward function

```bash
python reward_distillation.py \
  --env_name=antmaze-giant-navigate-v0 \
  --agent=agents/rew.py \
  --reward_shaping=1 \
  --fp_restore_epoch 2000000 \
  --project=rew_fn \
  --future_restore_path path/to/save_occupancy_model_dir \
  --batch_size 256 \
  --train_steps 2000000 \
  --state_action=True
```

### 3) Train GCIQL with ORS reward shaping

```bash
python main.py \
  --env_name=antmaze-giant-navigate-v0 \
  --seed=0 \
  --run_group=ORS \
  --eval_episodes=50 \
  --agent=agents/gciql.py \
  --fp_restore_epoch 2000000 \
  --project=antmaze-giant-navigate \
  --batch_size 1024 \
  --rew_restore_path path/to/reward_fn \
  --rew_restore_epoch 2000000 \
  --agent.expectile=0.6 \
  --use_rew_model=True \
  --agent.alpha=0.1 \
  --reward_shaping=1 \
  --train_steps=6000000 \
  --state_action=True \
  --agent.discount=0.995
```

## Notes

- Change the dataset by setting `--env_name`.
- See `occupancy_reward_shaping/impls/agents/` for agent-specific hyperparameters.
- For full hyperparameter lists, refer to the paper appendix.

## Repo Layout

- `occupancy_reward_shaping/impls/`: Training entrypoints and scripts.
- `occupancy_reward_shaping/impls/agents/`: Agent configs and defaults.
- `occupancy_reward_shaping/ogbench/`: OGBench environments and utilities.
