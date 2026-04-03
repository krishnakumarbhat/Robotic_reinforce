# pushcube_ppo

Task: `PushCube-v1`
Algorithm: `PPO`
Status: planned

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo pushcube_ppo
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

PPO should be stable on `PushCube-v1`, but it will likely require more environment interaction than SAC to match the same success rate.