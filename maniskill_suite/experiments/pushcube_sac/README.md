# pushcube_sac

Task: `PushCube-v1`
Algorithm: `SAC`
Status: planned

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo pushcube_sac
```

## Pilot Plan

- observation mode: state
- demo source: none
- budget: 500K timesteps
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

SAC should be the best RL-only benchmark for `PushCube-v1` because the task is continuous and strongly shaped by dense contact control.