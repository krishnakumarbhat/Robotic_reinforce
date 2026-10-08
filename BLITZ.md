# BLITZ — Kaggle pre-reset burn-down (43 h, resets Saturday)

Run in order after `go`. One kernel at a time (single remote-train slot). Each: push,
poll to terminal via `connect_gpu/run_gpu.sh` (or manual push + status), download outputs,
record below. STOP when sidebar counter hits 1 h reserve.

## Queue

| # | Kernel file | GPU? | Est | Purpose | Done |
|---|---|---|---|---|---|
| B1 | `experiments/kaggle_i6b_tags.py` | T4 ~1.5 h | quality-tag probe: [Q:SUCCESS] vs [Q:FAIL] vs none, 15 eps spatial, adapter step917 | ⬜ |
| B2 | `experiments/kaggle_i6_spatial2.py` (copy, N=2 spatial) | T4 ~1 h | fill the missing spatial cell (adapter+base), both timed out at N=10/4 | ⬜ |
| B3 | `experiments/kaggle_i12_cpu.py` (rig copy, 200 seeds) | CPU (no GPU quota) | I12 hard-regime matrix at 3cm/6° + noise sweep for re-validation | ⬜ |
| B4 | FT continuation (needs prep, ~8 h) | T4/T4x2 | expert FT more steps from step917 with fixed code | ⬜ prep only |
| B5 | I4 shortcut expert (needs new code) | A100/Colab | 1-NFE training per I4 spec | ⬜ loop owns |

## Push commands (from repo root; run_gpu handles metadata+poll+fetch)
```
GPU_BACKEND=kaggle KAGGLE_ACCELERATOR=NvidiaTeslaT4 ./connect_gpu/run_gpu.sh experiments/kaggle_i6b_tags.py
GPU_BACKEND=kaggle KAGGLE_ACCELERATOR=NvidiaTeslaT4 ./connect_gpu/run_gpu.sh experiments/kaggle_i6_spatial2.py
GPU_BACKEND=kaggle KAGGLE_ACCELERATOR= ./connect_gpu/run_gpu.sh experiments/kaggle_i12_cpu.py
```
Kernel files must stay self-contained (only code.py ships — never import repo modules).

## After reset
Saturday: fresh 45 h. Priority: B4 FT continuation, then I4, then I12-GPU variants.
