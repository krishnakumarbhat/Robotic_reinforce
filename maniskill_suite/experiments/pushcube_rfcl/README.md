# pushcube_rfcl

Task: `PushCube-v1`
Algorithm: `RFCL`
Status: planned

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo pushcube_rfcl
```

## Pilot Plan

- observation mode: state
- demo source: teleop
- budget: 5 teleop demos
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

RFCL should exploit a few push demonstrations more effectively than BC because it can recover online from small contact mismatches.