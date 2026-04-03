# pickcube_act

Task: `PickCube-v1`
Algorithm: `ACT`
Status: planned

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo pickcube_act
```

## Pilot Plan

- observation mode: state
- demo source: teleop
- budget: 3 teleop demos
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

ACT should be more tolerant than BC to timing differences across the same pick behavior.