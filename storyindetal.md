# The Full Story: From RL Sim to Open VLAs (Mar 2026 → Sep 24, 2026)

A detailed chronological record of everything that happened in this project —
every phase, every number, every failure and what it taught. Sources: git log
(97 commits), `experiments/worklog.md` (811 lines), `autoresearch_research.jsonl`
(66 rows), `results/`, `connect_gpu/outputs/`, and 29 Kaggle kernels (v8–v36).

---

## Prologue: RL Simulation Roots (Mar 6 → Jun 1, 2026)

| Date | Commit | What |
|---|---|---|
| Mar 6 | `856eedc` Initial commit | Repo born: README, gitignore, CI/CD |
| Mar 6 | `40ce067` | Complete RL simulation |
| Apr 3 | `d968296` | Imitation learning experiments |
| Jun 1 | `fdf8c99` | README image |

Three quiet months. RL simulation + imitation learning foundations, no robots yet.

## Act 1: The Autoresearch Loop Era (Sep 17 → Sep 20, 2026)

Sep 17: robot SO-arm scripts land, DP-Flow baseline + panda-gym smoke test.
Then the autonomous research loop ignites: **49 commits on Sep 18, 31 on Sep 19**
— 144 driver iterations exploring flow-matching/CFM/diffusion architectures
(N-series N1→N71), each with math proofs (`equations.md`), novelty searches,
and synthetic validation.

**JSONL verdicts (66 rows):** 39 validated-candidate, 10 discarded,
8 validated-candidate-predicted, 1 keep, 2 merge-under-36, 1 characterized.
Champion rule: beat-max over N40 (77.5 est) and N28 hard-10% (77.8 logic);
N71 reached **90.3 synthetic-predicted** — never confirmed on real physics.

**The honest core finding (Kernels B→E, ManiSkill, real contact dynamics):**

| Kernel | Scripted | Learned (CFM raw) | Gated | Verdict |
|---|---|---|---|---|
| B/C/D combined | ~0.81–1.0 | ~0.05–0.10 | ~0.17–0.375 | gate positive 4/4 dispatches |
| E (16 fresh seeds) | **0.8125** | 0.0625 | 0.1875, veto 0.40 | NOTHING beats scripted |
| F novel-dynamics | 0.625 novel vs 0.50 canon | 0.0625 | +0.25 canonical | scripted adapt gap **−0.125** (robust) |
| BC demos k12/24/48 | — | **~0** | — | proposer ~0 on both physics engines |

Metric priority set Sep 20: **learn faster 1st, adapt fluently 2nd**,
novelty 3rd/4th. No BEST promotion on synthetic estimates alone.
N71's 90.3 stands unconfirmed to this day — cited, never trusted.

**Lesson of Act 1:** the demo-anchored scripted core clears every bar;
every learned proposer fails. The gate helps (+0.08–0.375). The research
program pivots from "better architecture from scratch" to "stand on
open foundation models + keep the gate."

## Act 2: The VLA / Cloud Era (Sep 20 → Sep 22, 2026)

### Small/medium/big plan
Small = RTX 3050 4GB inference only. Medium = Colab/Kaggle post-train.
Big = cloud RL + async policy server. Prior work preserved as fallback,
never rewritten.

### SmolVLA track (all real, all logged)
- **Small smoke ✅:** SmolVLM2-500M INT8 — 460M params, 608MB alloc,
  7.2ms/step on the RTX. Verdict FIT with ~2.9GB headroom.
- **v16 (2k steps, T4):** loss 0.435, 1.79GB VRAM. Fixes en route: `av`+`num2words`
  missing, base config `push_to_hub:true/repo:null`, 1-cam vs 3-cam
  (solved via rename_map after reading lerobot's subset validator).
- **v19 (20k steps):** full run, ckpt → `krishnah27/smolvla-pusht-v20k`.
- **PushT evals:** base 0.060 max-reward → tuned-20k 0.26 (4x!) → 15k×b8
  **0.368**, success 0.0 throughout. Diagnosis: 80x under paper sample
  budget — learning, not converged.
- **v21 PIQ (FastVLA 4-bit, 6.4k steps):** loss 288→219, adapter on Hub.
- **G3 LIBERO contender (5k→15k steps):** loss 0.51→0.41, ckpt on Hub.

### The 35% number
G3E probe (Kaggle): **LIBERO-spatial 7/20 = 35.0%** (videos + eval_info
verified). Reference rows: pi0.5 98.8%, GR00T-N1.7 97.65% (both 3B,
paper-scale budgets).

### Model scouting (all verified, none assumed)
- **pi0.7: CLOSED.** HF searches empty, openpi stops at pi0.5. Newest
  trainable PI model = pi0.5-2.7B (22.5GB LoRA floor → paid tier).
- **pi0.6 exists** (Nov 2025, Gemma3-4B+860M) — card only, no open weights.
- **GR00T-N1.7** (Qwen3-VL, Apache-2.0-ish): suite = 6.9GB, inference fits
  T4, training needs 40GB+. Official LIBERO 97.0.
- **MolmoAct2** (Apache 2.0, fully open): 5B/~10GB — needs 24GB card.
  LeRobot can't load its sharded ckpt (single-file loader only); correct
  entry `AutoModelForImageTextToText` identified. Its stated weakness #1
  (no mid-batch reaction) is exactly what our async chunk-fusion fixes.
- **OpenVLA-7B:** loads marginal on T4 (15.06/15.6GB) BUT reports 901M
  params, not 7B — identity unconfirmed, flagged not trusted.
- **Gates:** gemma ✅ (user click), llama ✅ (user click). All public
  weights downloadable.

### Infrastructure won the hard way
- `run_gpu.sh` Kaggle flag fix; kernel outputs persist to repo immediately
  (never trust /tmp — reboot wiped it twice); train_full.txt on every burn;
  smoke dirs isolated from train dirs (the guard caused two crashes);
  client timeouts matched to session caps; streaming per-episode results.

## Act 3: Local Max-Out (Sep 23 → Sep 24, 2026)

Kaggle quota hit 30.00h (push rejected — $0 to learn). Colab died 5 times
(volatility rules written: micro-jobs only). The RTX became the eval lane:

**venv rodeo (all root-caused):** `PIP_TARGET` poisoned every pip (incl. the
venv's); transformers 5.x/4.x shadow fight (fixed 4.57.6); NTFS broke `regex`
(bypassed via `~/.cleansite` + path priority); CUDA-12 shim for xformers
(then uninstalled it — guarded import); new-ckpt config compat (iterative
key-drop); chat-template 404 shim (namespace sweep); kaggle-path override;
bool masks; CHW batches; CUDA placement. Clean ext4 venv now holds
torch 2.14 + lerobot 0.4.4 + robosuite 1.4.0 + mujoco 3.1.6.

**Local results (zero quota):**
| Experiment | Result | Time |
|---|---|---|
| v15k forward pass (own tuned ckpt) | 450M params, 950MB, **1.8ms/call**, sane outputs | 16s |
| LIBERO screening (custom driver) | **7/30 = 23.3%** (2/10, 2/10, 3/10) | 13 min |
| PushT v15k (matches Kaggle G2) | 0/20 in 102s | 2 min |
| INT8 projection | 906MB fp16 → ~678MB mixed; real INT8 VLM 608MB | 20s |

**Key local findings:** eval-CLI OOMs where standalone loads fine
(fragmentation → per-suite subprocesses); env defaults to pixels-only
(must pass `obs_type=pixels_agent_pos` or no state key exists); EGL costs
only ~724MB (fits); 100-ep protocol costs ~15h/suite (paid-tier only).

## Scoreboard (everything, one table)

| Claim | Number | Status |
|---|---|---|
| Scripted core, real physics | 0.81–1.0 | ✅ beaten by nothing |
| Gate over raw proposer | +0.08–0.375, 4/4 | ✅ |
| Scripted under novelty | 0.625 vs 0.50 (gap −0.125) | ✅ robust |
| BC from demos | ~0 at k12/24/48 | ✅ (negative result) |
| N71 synthetic | 90.3 | ⚠️ unconfirmed, cited only |
| SmolVLA PushT reward | 0.060 → 0.26 → 0.368 | ✅ learning curve |
| SmolVLA LIBERO-spatial | 35% probe / 23.3% screening | ✅ first numbers |
| SOTA reference | 96.85–97.65% @3B paper-scale | 🎯 target, not ours |
| Local inference | 1.8ms/call, 950MB | ✅ shippable |
| Hub assets | 4 tuned ckpts (v20k, PIQ-6k, v15k-b8, g3-10k) | ✅ safe |

## What remains (the buy-phase list)

1. GR00T-official LIBERO baseline (needs quota reset or paid).
2. LIBERO-PRO perturbation matrix (ours + officials).
3. PIQ rerun to full 10k.
4. pi0.5-LoRA vs official on LIBERO-PRO (RunPod 4090 $0.34/hr, ~$3–10).
5. Full-protocol 100-ep evals (~15h/suite — paid only).
6. Real-hand runs (blocked on hardware).

Total to finish: ~20–25h GPU ≈ **$12–17** (Colab Pro $11.99 doubles Kaggle to
60h/wk; Vast 4090 spot ~$0.15–0.26/hr beats RunPod's $0.34). Or $0 on the
free trickle (reset + local) over ~2 weeks.

## Appendix: 7-hour local research blitz (Sep 24, zero quota)

1. **Failure taxonomy:** failed LIBERO episodes show competent reach→grasp→lift
   on video while reward stays 0.0 — LIBERO reward is effectively binary;
   the dense signal it lacks is exactly what the gate provides.
2. **Run-variance proof:** same task + same seed → 3/6 successes; three full
   30-ep sweeps identical seeds → 7/30, 8/30, 9+14 pattern. Single-run LIBERO
   is a lottery; report distributions, always multi-run.
3. **Gate-as-predictor:** veto max_jerk>0.618 → prec 0.84 / rec 1.00 / F1 0.91
   (n=30, post-hoc). Held-out seeds: 0.58/0.88 — generalizes directionally;
   fixed thresholds don't survive; ADAPTIVE percentile veto is the right design.
4. **Local training verdict:** FastVLA-4bit inference FITS 4GB (3.64GB) but
   QLoRA training crashes (meta-device offload) — inference yes, training no.
5. **Paper skeleton** at `papers/gated_degradation_skeleton.tex` with all
   five claims and honest limitations.
