# Runtime Notes

This report records the environment-specific findings and compatibility fixes discovered while turning the ManiSkill scaffold into a working local experiment suite.

## Machine Validation

- Python: `3.10.12`
- GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`, `4096 MiB`
- Driver: `580.126.18`
- Vulkan: available and functional enough for ManiSkill smoke tests

## Package Resolution Notes

- ManiSkill `3.0.0b22` currently wants `gymnasium==0.29.1`, not `gymnasium>=1.0`.
- The official baseline entrypoints import `torch.utils.tensorboard.SummaryWriter`, so `tensorboard` is part of the practical baseline dependency set.
- The base suite runs through the local `.venv` interpreter directly instead of depending on shell activation.

## Data Layout Notes

- `teleop/` and `motionplanning/` downloads expose the simple `trajectory.h5` plus `trajectory.json` layout.
- Official RL downloads for `PickCube-v1` are stored as control-mode-specific files such as `trajectory.none.pd_joint_delta_pos.physx_cuda.h5` with matching `.json` files.
- The suite replay and command-generation scripts were updated to resolve that RL naming scheme automatically.

## Hybrid Baseline Compatibility Fixes

Two concrete compatibility issues blocked RFCL and RLPD and were fixed locally in this workspace:

1. ManiSkill `flatten_state_dict` assumed recursive results always had a torch-style `nelement()` method. Replay datasets backed by NumPy arrays triggered `AttributeError`. The local ManiSkill install and the cloned ManiSkill source were patched to treat empty NumPy arrays via `.size == 0`.
2. The editable `rfcl_jax` package still used deprecated `jax.tree_map` calls. These were updated to `jax.tree_util.tree_map` for compatibility with the installed JAX release.

## JAX Runtime Note

- The installed JAX stack runs, but it logs a CPU fallback warning because a CUDA-enabled `jaxlib` wheel is not present in this environment.
- RFCL and RLPD therefore execute correctly here, but not on GPU.

## Current Pilot Outcomes

- `pickcube_bc`: finished a 200-iteration pilot on 3 replayed teleop demos, loss dropped, task success stayed at `0.0`.
- `pickcube_sac`: finished a 10k-step pilot, achieved nonzero return (`3.74`), success stayed at `0.0`, and hit a teardown-only simulator exception after completion.
- `pickcube_rfcl`: completed a short 5-demo hybrid run end to end after compatibility patches, but success and return stayed at `0.0`.
- `pickcube_rlpd`: completed a short 20-demo offline-to-online pilot on replayed RL data, but evaluation success and return stayed at `0.0`.

The aggregate table is regenerated in `reports/experiment_results.md`.

## Remaining Baseline Blockers

- Official PPO currently still needs a clean re-bootstrap of the local environment after the `PYTHONPATH` hardening change. The isolated venv path exposed missing transitive `tensorboard` dependencies in the older environment state, which indicates the current `.venv` should be rebuilt rather than patched incrementally.
- ACT currently stops at import time because `torchvision` is not yet installed in the suite environment.
- Diffusion Policy currently stops at import time because `diffusers` is not yet installed in the suite environment.
- PushCube and StackCube combinations remain scaffolded and documented but were not executed in this session.