# pickcube_rlpd

Task: `PickCube-v1`
Algorithm: `RLPD`
Status: pilot_complete

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo pickcube_rlpd
```

## Pilot Plan

- observation mode: state
- demo source: official RL demos
- budget: 100 demos pilot
- control mode: `pd_joint_delta_pos`

## Metrics To Record

| Field | Value |
| --- | --- |
| status | pilot_complete |
| success_once | 0.0 |
| success_at_end | 0.0 |
| return | 0.0 |
| train_steps | 2000 |
| wall_clock_minutes | 30.3 |
| notes | 20 replayed RL demos used; 2k-step sample-efficient pilot finished with zero eval success and zero return; suite RL replay path had to be fixed for official `trajectory.none.*` downloads |

## Hypothesis

RLPD should become competitive on `PickCube-v1` only when it is given enough prior data to behave like a real offline-to-online method.

## Latest Pilot

The first automated pilot used `20` official `PickCube-v1` RL demonstrations replayed into `trajectory.state.pd_joint_delta_pos.physx_cpu.h5`, then ran RLPD with a shortened sample-efficient config for `2,000` environment steps.

Observed outcome:

- dataset loading reported `20` demos and `930` frames
- evaluation at the end of the pilot logged `success_once = 0.0`
- evaluation at the end of the pilot logged `success_at_end = 0.0`
- evaluation return stayed at `0.0`
- wall-clock time from TensorBoard was about `30.3` minutes
- JAX ran on CPU because a CUDA-enabled `jaxlib` wheel was not installed in this environment

Interpretation:

The offline-to-online pipeline now executes correctly on local RL prior data, but this short CPU-JAX pilot is far too small to show the advantage RLPD is designed for. It remains a later-stage method unless you commit a larger prior dataset and a longer budget.

## Pilot Note

The official ManiSkill RL download layout does not use `trajectory.h5`; it stores control-mode-specific files like `trajectory.none.pd_joint_delta_pos.physx_cuda.h5`. The suite scripts were updated to resolve that format automatically.