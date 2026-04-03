# stackcube_ppo

Task: `StackCube-v1`
Algorithm: `PPO`
Status: planned

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo stackcube_ppo
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

PPO on `StackCube-v1` will be a stability reference, but it is unlikely to be sample-efficient enough to solve stacking quickly on a small machine.