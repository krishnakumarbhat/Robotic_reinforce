# MISSION-15H: 15-hour breakthrough run (prepended every iteration via AUTORESEARCH_DIRECTIVE)

## LINE 0 — GROUNDING (do this before anything else, every iteration)
- `pwd` MUST print `/media/pope/projecteo/github_proj/a_resume/Robotic_reinforce`. If not, `cd` there first.
- ALL repo files resolve under that absolute path. NEVER guess `/home/pope/<repo-file>` — that path does not contain this project and any Read there fails. (Root-caused 2026-09-18: process cwd was always correct; the agent hallucinated `$HOME`-anchored absolute paths and burned a full iteration. This anchor prevents recurrence in ANY project using this driver: always state the absolute project path up front.)
- Local validation interpreter: `PYTHONPATH="<repo>/panda-gym:$PYTHONPATH" python3.10` (APPEND, never replace — base PYTHONPATH carries GPU libs).

## Consolidated context — ALL history kept, nothing pruned (user directive)
- Branch `research/dp-flow-2026-09-17`, 54 commits. JSONL 37 rows (30 validated-candidate, 3 discarded, 1 keep, 1 merge-under-36). Worklog 36 runs, segments 5–13 + stall-break. Graph 35 nodes/34 edges. Equations rows to 36. Strategies 44 rows.
- Driver iters reached 144, stage checkpoints to 148. Dashboard file is STALE (shows only runs 1–4) — regen it before trusting any summary; JSONL is source of truth.

## Metric priority (user order 2026-09-20 — novelty is 3rd/4th, architecture must WIN on these)
- 1st LEARN FASTER: episodes/demos/wall-time to success threshold (k-demo ablations, data-efficiency curves). A better architecture reaches criterion with less data/time.
- 2nd EXPERIMENT BETTER + ADAPT FLUENTLY: real-physics success on contact-rich tasks + adaptation gap (novel tool/scene vs canonical after 3s demo/few-shot) + recovery latency. Beat pi0/pi0.5/FAST-class and GR00T-N1.5-class behavior on THESE axes, not on synthetic cosmetics.
- 3rd/4th novelty + SOTA-table cosmetics. No BEST promotion on synthetic estimates alone; validated-candidate-predicted stays until real-physics confirmation.

## Champion rule — beat-max(all), keep rest for final test
- Bar to beat: **N40 77.5 est AND N28 hard-10% 77.8 logic**; N31 77.1 synthetic stays unconditional fallback.
- Keep ALL validated-candidates for the final comparative test (N34 76.8, N33 76.45, N28/N29/N32 76–76.2, N39 72.0, N41 73.0). Never delete history.
- Dead ends — never retry, always cite: N9/N10 β-collapse (kernel→uniform, gain 0), N30 lift −0.10 (collapse confirmed), N33 adaptive −0.65 (nested-loop oscillation), N17/N18 plateau 76.0 (gradient clash).

## 500-run matrix (approved) — complex sims ONLY, no simple pick-and-place
- Policies: ACT · Diffusion 16/32/64-step · CFM 5/10-step · physical prompting (3–12s) · DP-Flow N31/N40/N28/N41.
- Scenarios: Tier 2 illumination/shading → Tier 3 clutter/deformable → Tier 4 zero-shot tool-swap (banana-as-wipe, cardboard-as-broom). Procedural generator `benchmarks/env_generator.py` (build it if missing).
- Metrics per cell: success, contact-force stability, trajectory jerk (L2), recovery latency → `results/metrics.json` ONLY.

## Disk policy (`/` is 81% full — strictest constraint)
- Procedural rollouts + streaming replay ONLY. No HDF5/Zarr dumps. Purge `tmp/`, `.cache/`, video after every batch.
- Keep top-2 checkpoints FP8/INT8/BF16 + scalar logs. Checkpoints stay on remote (Kaggle/Colab/HF) until final.

## GPU budget + schedule (15h wall-clock)
- Local RTX 3050 (3.7G free now): smoke + INT8 + batch≤8 via `/tmp/.opencode_gpu.lock`. Small prototyping/MuJoCo only.
- Kaggle (15h continuous): ManiSkill validation queue for top candidates (N40, N31, N28, N41) — the missing `keep ≥70` confirmation. `GPU_BACKEND=kaggle` via `connect_gpu/run_gpu.sh`. Save checkpoints to Drive/HF every N steps; sessions die randomly.
- Colab (max 4–5h): single heaviest job only — full 3s-demo calibration + contact dynamics on champion challenger. Tear down with `colab stop` when done. Check quotas first.
- Monitor every remote job each iteration: if a model/job misbehaves (OOM, stall, crash) → downscale batch, gradient checkpointing, FP16/INT8, or CPU offload; patch, unit-test fix, resume. Never leave a broken job burning quota.

## Models — subagents of self only, 1 then 2 on rate-limit
- Worker = default session model (proven). Brain relay seat = DeepSeek-V4-Flash (hallucinates: see iter-144 `Session not found`).
- DeepSeek hallucination guards (mandatory): NO keep without (a) existing evidence artifact path, (b) numerically re-run math, (c) real novelty-search hits, (d) metric recomputed from evidence. Fail any → discard + log. Rate-limit/5xx → cascade once per frozen directive, never stall.

## Supervisor (self-copy) checks every iteration
Stall (3 flat iters) → rotate category + frontier node (deepest gap N5/N2 divergence 0.53). Regression vs champion → instant revert. Entropy dominance >0.5 / gradient clash → divergence flag. Idea drought → inject lateral frontier.

## Paper target
CoRL 2026 primary (`papers/templates/neurips.tex`, arXiv-first per `citation_playbook.md`). Every keep needs Theorem/Lemma + proof + experiment vs SOTA table. Citations real (verify S2/arXiv).

## LIT-100 track (read up to 100 field papers, harvest tasks+data)
- Target: 100 notes in `papers/notes/<slug>.md` (frontmatter: title/arxiv/venue/citedByCount/mechanisms/cracks + ≤30-line body: claim, method, tasks+data solved, gaps). One `research/literature_matrix.md` row per paper.
- Scope: VLA/flow-matching/diffusion/ACT/tool-use/manipulation 2024–2026 (seed set already covered: pi0 v4, ProgressVLA, π0/FAST/0.5 survey — extend outward, newest first).
- Fetch via `~/.config/opencode/scripts/autoresearch-fetch.sh` (cache in `webcache/`); tiny text artifacts ONLY (no PDFs/datasets locally).
- VERIFY-BEFORE-WRITE (burned 2026-09-18: harvest batch shipped 6 fake arXiv IDs + 5 unverified): every arXiv ID must resolve via Semantic Scholar `api.semanticscholar.org/graph/v1/paper/arXiv:<id>?fields=title` (fallback: export.arxiv.org with browser UA) BEFORE the note is written. Mismatch → quarantine to `papers/notes/quarantine/`, never matrix. Near-duplicate slugs of seed notes → drop. Unverified-on-rate-limit → quarantine + matrix `UNVERIFIED` section, recheck later.
- Every harvest batch ends with: new gaps → new frontier nodes in strategy graph. Papers that solve our gap kill the idea (discard + cite).

## MULTI-PATH TREE (local never idles; cloud trains, local infers)
- Each kept idea spawns N architecture children (vary ONE axis each: attention / projection / conditioning / calibration). Different types, trained in parallel.
- Cloud (Kaggle queue + Colab single-heavy): full training per child. New wacky idea → new child branch, never rewrite champion in place.
- Local RTX 3050 queue (highest priority first, via `/tmp/.opencode_gpu.lock`, batch≤8, FP16/INT8): (1) current-iter smoke/validation, (2) INT8 inference of cloud-returned checkpoints, (3) panda-gym/MuJoCo rollout ablations, (4) math verification. If queue empty → pull next cloud checkpoint for local inference. Idle GPU = protocol violation, log it.
- Cloud return protocol: verify (recompute metric from artifact + bounded checks) → compare vs beat-max + fresh-paper check → keep/discard/merge → spawn next N children or rotate.

## DISK GUARD + HOURLY MONITOR (`monitor-15h.sh`, auto-started)
- Every hour logs driver/workers/jsonl-count/disk/GPU to `MONITOR-15H.md`. If `/` avail < **5GB** → runs `autoresearch-loop.sh stop` immediately and logs the trip (crash protection). I review this log each check-in = the hourly build-mode review.
