# pickcube_sac

Task: `PickCube-v1`
Algorithm: `SAC`
Status: pilot_complete

## Run Command

```bash
python maniskill_suite/scripts/run_from_matrix.py --combo pickcube_sac
```

## Pilot Plan

- observation mode: state
- demo source: none
- budget: 500K timesteps
- control mode: `pd_ee_delta_pos`

## Metrics To Record

| Field | Value |
| --- | --- |
| status | pilot_complete |
| success_once | 0.0 |
| success_at_end | - |
| return | 3.74 |
| train_steps | 10048 |
| wall_clock_minutes | 15.4 |
| notes | 10k-step pilot reached nonzero return but no success; ManiSkill/SAPIEN raised a cleanup TypeError on shutdown after training completed |

## Hypothesis

SAC should be the strongest RL-only baseline on `PickCube-v1` under a 4 GB state-based setup.

## Latest Pilot

The first automated pilot ran SAC on `PickCube-v1` in state mode with `pd_ee_delta_pos`, `8` training environments, `4` evaluation environments, a CPU replay buffer, and `10,000` target timesteps.

Observed outcome:

- final reported `success_once` was `0.00`
- final reported return was `3.74`
- the run completed the short training budget in about `15.4` minutes
- a `TypeError: 'NoneType' object is not callable` was raised during environment teardown after training finished

Interpretation:

The RL baseline is already extracting shaped reward signal under the small-budget setup, which is more promising than the tiny-demo BC pilot. The teardown exception appears to be a cleanup issue in the simulator stack rather than a training-time failure.