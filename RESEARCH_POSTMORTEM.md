# RESEARCH POSTMORTEM — pre-2026-10-06 runs (append-only; nothing removed)

## Where the program stands (all evidence in repo, see experiments/worklog.md)
- **Sim track (KEPT, paper core):** trochoid champion B 20/20 p=0.0083; I9 depth-reg
  confirmed n=100 (audited); I2 conformal tag B 1.00 vs 0.75; I23 reg×centring;
  translation dose-response A-breaks-(6,9]cm / B-breaks-(4.5,6]cm; yaw 6° costs
  ~2-4pts; graceful coverage slope (binary gate = threshold artifact).
- **VLA track (BURIED 2026-10-05, v68):** step917→step10607, 13× steps, loss
  .55→.28, still LIBERO 0.0 = base 0.0 on all 6 cells. FT-for-LIBERO dead until a
  >floor adapter exists. Self-contained merged weights on Hub
  (`krishnah27/smolvla-aegis-ft-step10607` + `model.safetensors`).
- **Instruments kept:** I6b static tag-conditioning probe (semantic_ratio 2.16,
  repeat_gap 0.0); `merge_lora_cpu.py`; `kaggle-watch.sh`; `reap_orphans()`.
- **Invalidated (do NOT resurrect):** 2× I9 fabrication, run-169 (soft gate +
  rogue seg), G3 family (106+ replay refusals logged), FACC/SE3/EBM family.
- **Infra:** Kaggle ~22h left (resets Sat); Colab Pro + reserve rule; RTX loop
  evals; disk 56% (pruned 1433 + 4045 sessions, archives in session_archive/).

## Open edges for the new run
N198 (director-owned), N478d, I4 1-step expert, I3 PPO closeout, I11, I8-retry,
paper (LaTeX + arXiv), A100 full run ONLY if a >floor adapter appears.

## New mandate 2026-10-06 (director): stay on goal, append-only
Goal = unified edge-deployable adaptive manipulation (directive below). New
segment appends; no state removed. Rules added: G7 stay-on-goal, G8 append-only.
