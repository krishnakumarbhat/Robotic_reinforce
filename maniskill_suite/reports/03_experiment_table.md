# Experiment Table

| Combo | Task | Algorithm | Family | Demo Source | Pilot Budget | Folder |
| --- | --- | --- | --- | --- | --- | --- |
| `pickcube_ppo` | PickCube-v1 | PPO | online RL | none | 1M steps | `experiments/pickcube_ppo` |
| `pickcube_sac` | PickCube-v1 | SAC | online RL | none | 500K steps | `experiments/pickcube_sac` |
| `pickcube_bc` | PickCube-v1 | BC | IL | teleop | 3 demos | `experiments/pickcube_bc` |
| `pickcube_act` | PickCube-v1 | ACT | IL | teleop | 3 demos | `experiments/pickcube_act` |
| `pickcube_diffusion_policy` | PickCube-v1 | Diffusion Policy | IL | teleop | 3 demos | `experiments/pickcube_diffusion_policy` |
| `pickcube_rfcl` | PickCube-v1 | RFCL | online demos | teleop | 5 demos | `experiments/pickcube_rfcl` |
| `pickcube_rlpd` | PickCube-v1 | RLPD | online demos | official RL demos | 100 demos pilot | `experiments/pickcube_rlpd` |
| `pushcube_ppo` | PushCube-v1 | PPO | online RL | none | 1M steps | `experiments/pushcube_ppo` |
| `pushcube_sac` | PushCube-v1 | SAC | online RL | none | 500K steps | `experiments/pushcube_sac` |
| `pushcube_bc` | PushCube-v1 | BC | IL | teleop | 3 demos | `experiments/pushcube_bc` |
| `pushcube_act` | PushCube-v1 | ACT | IL | teleop | 3 demos | `experiments/pushcube_act` |
| `pushcube_diffusion_policy` | PushCube-v1 | Diffusion Policy | IL | teleop | 3 demos | `experiments/pushcube_diffusion_policy` |
| `pushcube_rfcl` | PushCube-v1 | RFCL | online demos | teleop | 5 demos | `experiments/pushcube_rfcl` |
| `pushcube_rlpd` | PushCube-v1 | RLPD | online demos | official RL demos | 100 demos pilot | `experiments/pushcube_rlpd` |
| `stackcube_ppo` | StackCube-v1 | PPO | online RL | none | 1M steps | `experiments/stackcube_ppo` |
| `stackcube_sac` | StackCube-v1 | SAC | online RL | none | 500K steps | `experiments/stackcube_sac` |
| `stackcube_bc` | StackCube-v1 | BC | IL | teleop | 3 demos | `experiments/stackcube_bc` |
| `stackcube_act` | StackCube-v1 | ACT | IL | teleop | 3 demos | `experiments/stackcube_act` |
| `stackcube_diffusion_policy` | StackCube-v1 | Diffusion Policy | IL | teleop | 3 demos | `experiments/stackcube_diffusion_policy` |
| `stackcube_rfcl` | StackCube-v1 | RFCL | online demos | teleop | 5 demos | `experiments/stackcube_rfcl` |
| `stackcube_rlpd` | StackCube-v1 | RLPD | online demos | motionplanning demos | 100 demos pilot | `experiments/stackcube_rlpd` |

## Read This Table Correctly

- The teleop pilot budgets are for a first pass, not a publication-grade dataset.
- The RLPD rows assume larger replay data than 2 to 3 teleop demos.
- `StackCube-v1` is included to stretch the pipeline, not because it is the safest first run.