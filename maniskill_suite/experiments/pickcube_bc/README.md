# pickcube_bc

Task: `PickCube-v1`
Algorithm: `BC`
Status: pilot_complete

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo pickcube_bc
```

## Pilot Plan

- observation mode: state
- demo source: teleop
- budget: 3 teleop demos
- control mode: `pd_ee_delta_pos`

## Metrics To Record

| Field | Value |
| --- | --- |
| status | pilot_complete |
| success_once | 0.0 |
| success_at_end | 0.0 |
| return | 0.0 |
| train_steps | 200 |
| wall_clock_minutes | - |
| notes | 3 official teleop demos replayed to state; 200-iteration pilot never achieved success in any 8-episode evaluation checkpoint |

## Hypothesis

BC should solve short, clean pick trajectories if the 3 teleop demos are consistent, but it will likely fail under state drift.

## Latest Pilot

The first automated pilot used the official `PickCube-v1` teleop demonstrations, replayed them into `trajectory.state.pd_ee_delta_pos.physx_cpu.h5`, and trained BC for 200 iterations with evaluation every 50 iterations.

Observed outcome:

- loss fell from about `0.2601` to about `0.0085`
- `success_once` stayed at `0.0`
- `success_at_end` stayed at `0.0`
- average return stayed at `0.0`

Interpretation:

The policy fit the action distribution numerically, but it did not convert that fit into task success. This is a useful negative baseline for the tiny-demo setting and supports the expected covariate-shift failure mode.