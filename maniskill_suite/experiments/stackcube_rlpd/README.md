# stackcube_rlpd

Task: `StackCube-v1`
Algorithm: `RLPD`
Status: planned

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo stackcube_rlpd
```

## Pilot Plan

- observation mode: state
- demo source: motionplanning
- budget: 100 demos pilot
- control mode: `pd_joint_delta_pos`

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

RLPD on `StackCube-v1` is only worth the cost if a larger, cleaner prior dataset is available, which makes it a later-stage comparison instead of a first pilot.