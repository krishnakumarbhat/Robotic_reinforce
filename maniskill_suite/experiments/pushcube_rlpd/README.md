# pushcube_rlpd

Task: `PushCube-v1`
Algorithm: `RLPD`
Status: planned

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo pushcube_rlpd
```

## Pilot Plan

- observation mode: state
- demo source: official RL demos
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

RLPD should benefit `PushCube-v1` if the official prior data is large enough, but it is not the right algorithm for a tiny teleop pilot.