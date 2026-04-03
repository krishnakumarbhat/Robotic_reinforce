# stackcube_rfcl

Task: `StackCube-v1`
Algorithm: `RFCL`
Status: planned

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo stackcube_rfcl
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

RFCL is the most credible few-demo method in this suite for `StackCube-v1` because it can turn tiny demonstration priors into online correction.