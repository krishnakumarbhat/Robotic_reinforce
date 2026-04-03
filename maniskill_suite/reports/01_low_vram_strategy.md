# 4 GB VRAM Strategy

This suite is intentionally biased toward feasibility on a small GPU.

## Hard Rules

- Use `obs_mode=state` first.
- Disable video capture during training.
- Use CPU replay for demonstrations unless you have already confirmed `physx_cuda` is stable.
- Keep SAC environment count modest.
- Treat ACT and Diffusion Policy as second-stage runs, not the first experiment of the day.

## Recommended Order

1. `PickCube-v1` with PPO or SAC in state mode.
2. `PickCube-v1` with BC from 3 teleop demos.
3. `PickCube-v1` with RFCL from 5 teleop demos.
4. `PushCube-v1` with SAC.
5. `StackCube-v1` only after the full pipeline works.

## Safe Defaults

### RL

- PPO: 64 envs, 1M timesteps, state only
- SAC: 16 envs, 500K timesteps, state only, small replay buffer

### IL

- BC: 3 replayed teleop demos for a pilot, 10K iterations
- ACT: 3 replayed teleop demos for a pilot, 20K iterations
- Diffusion Policy: 3 replayed teleop demos for a pilot, 30K iterations

### Demo-Aware Online Learning

- RFCL: 5 demos, sample-efficient config, CPU backend
- RLPD: 100 demo pilot if using official data, then scale to 1000 only if justified

## Things To Avoid First

- RGB or RGBD training before you have one working state baseline
- large replay conversion jobs with hundreds of parallel environments
- all-task sweeps at once
- official walltime-efficient settings copied from large GPUs without reduction

## JAX Note

RFCL and RLPD rely on the official ManiSkill demo-aware baselines, which use extra JAX ecosystem dependencies. Keep them out of the default bootstrap path so the base environment stays light and debuggable.