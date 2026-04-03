# pickcube_ppo

Task: `PickCube-v1`
Algorithm: `PPO`
Status: planned

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo pickcube_ppo
```

## Pilot Plan

- observation mode: state
- demo source: none
- budget: 1M timesteps
- control mode: `pd_ee_delta_pos`

## Metrics To Record

| Field | Value |
| --- | --- |
| status | planned |
| success_once | - |
| success_at_end | - |
| return | - |
| train_steps | 0 |
| wall_clock_minutes | 0 |
| notes | not run yet |

## Hypothesis

PPO should be the safest RL-only baseline for `PickCube-v1`, but it will likely trail SAC on sample efficiency.