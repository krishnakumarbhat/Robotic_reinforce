# pickcube_rfcl

Task: `PickCube-v1`
Algorithm: `RFCL`
Status: pilot_complete

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo pickcube_rfcl
```

## Pilot Plan

- observation mode: state
- demo source: motionplanning
- budget: 5 motion-planning demos
- control mode: `pd_joint_delta_pos`

## Metrics To Record

| Field | Value |
| --- | --- |
| status | pilot_complete |
| success_once | 0.0 |
| success_at_end | 0.0 |
| return | 0.0 |
| train_steps | 10000 |
| wall_clock_minutes | 0.16 |
| notes | 5 motion-planning demos replayed to joint-control state; both curriculum stages ran, but percent solved stayed at 0.0; required local ManiSkill and RFCL compatibility patches |

## Hypothesis

RFCL should outperform BC on very small demo counts by using online correction instead of pure supervised imitation.

## Latest Pilot

The first automated pilot used `5` official `PickCube-v1` motion-planning demonstrations replayed into `trajectory.state.pd_joint_delta_pos.physx_cpu.h5`, then ran RFCL with the sample-efficient config for a short `5,000`-step stage-1 pass followed by stage-2 seeding and online rollout.

Observed outcome:

- dataset conversion loaded `5` demos and `361` frames
- stage 1 completed its short training budget
- stage 2 started and loaded an offline buffer with `5000` interactions
- recorded `success_once` stayed at `0.0`
- recorded `success_at_end` stayed at `0.0`
- recorded return stayed at `0.0`
- logged wall-clock time was about `9.8` seconds
- the run completed only after fixing two compatibility issues: ManiSkill state flattening on NumPy-backed demo data and deprecated `jax.tree_map` calls in the editable RFCL package

Interpretation:

The hybrid pipeline now runs end to end on this machine, which is the important result. This short CPU-JAX pilot did not solve the task, but it established that RFCL can be executed locally once the current ManiSkill and JAX compatibility gaps are patched.