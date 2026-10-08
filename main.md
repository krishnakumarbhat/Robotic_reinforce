# main.md — Handoff for the next model (Project Aegis Sanitation)

Written 2026-09-27 by the director session. Read this file first, then:
`.autoresearch-directive-aegis.md` (binding rules G1-G7) and `autoresearch_research.ideas.md`
(idea queue v3 + specs). Repo: `/media/pope/projecteo/github_proj/a_resume/Robotic_reinforce`.

## 1. Goal
A plinth-mounted 6-DoF arm cleans commercial toilet fixtures and must transfer zero-shot to new
customer fixtures (shape, finish, ±15 cm placement, ±10° yaw, friction 0.05-0.80). Edge target:
≤500M params, ≤1.5 GB VRAM, ≤25 ms per action chunk. Target venue: CoRL / ICRA.
Only **physical** results count (PyBullet contact physics). The old synthetic "novelty_score" work
(N70-N76, eq78, "champion N74 92.3") is retired.

## 2. Current truth (all verified by execution)
Canonical rig: `experiments/kaggle_aegis_sweep.py`, run with the **system `python3`**
(pybullet is installed there, not in `~/.venvs/infer`). 20 seeds × 3 suites takes about 25 s on CPU.

| path (20 seeds, paired) | A succ | B succ | R succ (random new customers) |
|---|---|---|---|
| raster (scripted baseline) | 0.70 | 0.65 | 0.65 |
| rounded / spiral | no gain → discarded | | |
| **trochoid = CHAMPION** | 1.00 | **1.00** (Fisher p=0.0083 vs raster) | 0.95 |
| trochoid + pose noise 1 cm / 2° | 0.75 | 0.75 | 0.65 |

**UPDATE 2026-09-29 (I22 row-centring changed everything):** with rows at cell centres, the easy
regime (noise 0 AND 1cm/2°) is SOLVED — raster, trochoid and I9-registration all score 20/20.
Trochoid-vs-raster now ties (keep=FALSE). All future keep claims MUST be at 3cm/6° noise or harder.
The trochoid, I9-depth (run 125) and I2-tag keeps were measured pre-I22 and keep their mechanism
claims, but their NUMBERS must be re-validated at 3cm/6° on the current rig before paper citation.

Pose-noise curve (I5, trochoid, B): 1.00 → 0.75 (1 cm/2°) → 0.60 (2 cm/4°) → 0.50 (3 cm/6°).
**Robustness to fixture-pose error is the open frontier.**

Evidence: `results/aegis_v2/*.jsonl`. Episode success = `coverage_cont ≥ 0.90`. `coverage_cont` is
a footprint-based fine grid; the rig recomputes it from physics contacts at return time.

The old "Fixture-B 0.15" number was two measurement bugs (time-based phase labels and cell-boundary
quantisation), now fixed. Do not cite it.

Rig limits to state in the paper: force-driven pad with no arm, flat top faces, normal force ≈0.5 N
(not the 10-25 N spec), and the Tier-4 gate is only a post-hoc tag (not an in-loop interceptor yet).

## 3. Standard experiment command
```
timeout 1200 python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload \
  --path <candidate> --compare trochoid --suites fixture_A,fixture_B,fixture_R \
  --out results/aegis_v2/<idea>_<tag>.jsonl
# env knobs: AEGIS_POSE_NOISE="sigma_m,sigma_deg"  AEGIS_BASE_SEED=...
```
The `compare` record carries `keep`. Keep requires all of:
- ≥20 seeds
- B transfer_success > 0.70
- a significant win over the paired baseline: Welch p<0.01 on coverage **or** Fisher p<0.01 on success
- no regression in mean coverage

## 4. Idea queue (details in `autoresearch_research.ideas.md`)
Done: I0 (fine coverage, keep), I1 (trochoid KEEP; rounded/spiral discarded), I5 (pose-noise curve),
I10 (footprint pitch, DISCARD).

Next, in this order:
1. **I9 — DONE (KEEP, run 91, director).** Depth-sweep registration works: exclude tool/plane bodies
   from rays, bracket the top face, PCA yaw with pi-resolve, round keeps prior yaw. n=60 paired at
   noise 3cm/6°: R 44/60 vs 29/60 Fisher p=0.0085; B 44/60 vs 33/60 (Fisher p=0.056, coverage
   p=0.0004). Scope: helps iff pose error is large (at 1cm/2° the pad absorbs error: no gain).
   Evidence: `results/aegis_v2/i9_REG60.jsonl`, `i9_NOREG60.jsonl`. Two worker I9 "keeps" were
   audited INVALID (no-noise cited as reg; teleport+s fake bump).
2. **I7** — in-loop Tier-4 gate (veto → compliant retract 3 cm → re-engage). Always report gate on vs off.
3. **I2** — split-conformal gate threshold per friction bin.
4. **I8** — curved bowl surface (heightfield).
5. **I11** — normal-force regulation into the 10-25 N band.
6. **I3** — residual PPO on top of the champion.
7. GPU ideas: **I4** (1-NFE shortcut/MeanFlow expert to reach ≤25 ms; measured now 285 ms/chunk on
   RTX 3050) and **I6** (quality-tag conditioning).
8. **I12** — final 200-seed robustness matrix (paper table), run as a Kaggle **CPU** kernel.

## 5. GPU lane (Colab Pro + Kaggle 45 h/week)
- Always dispatch through `connect_gpu/run_gpu.sh`. It is a copy of the shared
  `/media/pope/projecteo/connect/connect_gpu/run_gpu.sh`; edit the shared one, then copy it over.
  - The Colab session is stopped by a shell trap, even if the job crashes (tested).
  - Each job gets a unique session name.
  - A traceback sets rc=1.
  - Kaggle jobs are polled until they reach a terminal state.
  - Every job appends a line to `connect_gpu/usage.log`.
- Colab accounts: `source /media/pope/projecteo/connect/connect_gpu/use-colab.sh pro|old|show`.
  Pro = claude.raseksha@gmail.com. Always pass `colab --config "$COLAB_CONFIG"`.
- HF token works: the project `connect_gpu/huggingface/.env` → `krishnah27`. The shared-folder copy is
  a placeholder; don't use it. Checkpoints go to the HF Hub only.
- Burst scripts: `experiments/colab_aegis_boot.py` is the launcher. It pins
  torch 2.10 / transformers 4.57.6 / peft 0.21.0 / hub 0.35.3 / safetensors 0.8.0, removes torchao,
  and runs the trainer in a subprocess. `experiments/colab_aegis_ft.py` is the trainer.
- Current mode is `AEGIS_NO_LORA=1`: frozen VLM plus full fine-tuning of the expert (99.9M trainable).
- STATUS 2026-09-27: A100 burst trained to ~step587, checkpoints on Hub
  (`krishnah27/smolvla-aegis-ft-step221/432/551/587/649/866`). Dataloader fix (parquet-direct action
  reads) committed after; a T4 worker run with current code is in flight.
- The earlier "RecursionError" was my debug monkeypatch (a wrapper stacked on every forward pass),
  not PEFT. The hooks are removed. LoRA could be re-enabled (`AEGIS_NO_LORA=0`), but that is untested.

### RUNNING NOW
An A100 burst started via `run_gpu.sh` (log `/tmp/a100_burst.log`). It is expert-only, pushes
checkpoints to `krishnah27/smolvla-aegis-ft-step*`, and stops itself after 3.5 h plus teardown.
It is working but **slow**: 207 steps in 32 min.

**Cause:** in `build_batch` (`colab_aegis_ft.py`), the H=50 action window calls `ds[i]` 50 times per
sample, and each call **decodes video frames**.

**FIX TO APPLY NEXT (not done yet):** read actions without decoding video:
```python
acts = torch.stack([torch.as_tensor(ds.hf_dataset[i]["action"]) for i in idxs])
```
(`ds.hf_dataset` is the parquet table with no video decoding; call `ds._ensure_hf_dataset_loaded()`
first.) Keep `ds[idxs[0]]` for the images. Expect about a 10-50× speed-up.

Also check the data pipeline stays within 16 micro-batches, then rerun the burst:
```
R=$PWD; GPU_BACKEND=colab COLAB_GPU_TYPE=A100 GPU_MAX_MIN=240 \
COLAB_UPLOADS="$R/connect_gpu/huggingface/.env:/content/aegis.env,$R/results/aegis_v2/i1_trochoid_vs_raster.jsonl:/content/aegis_context.json,$R/experiments/colab_aegis_ft.py:/content/aegis_run.py" \
COLAB_DOWNLOADS="/content/aegis_ft_report.json:$R/results/aegis_ft_report.json" \
./connect_gpu/run_gpu.sh $R/experiments/colab_aegis_boot.py
```
Only one GPU burst at a time. Check `colab --config "$COLAB_CONFIG" sessions` before booking another.

## 6. Autoresearch loop (unattended workers)
- Start the loop only with `./autoresearch-launch.sh`, detached:
  `setsid nohup ./autoresearch-launch.sh >/tmp/launch.out 2>&1 &`. What the launcher does:
  - probes the worker models (inkling / muse-spark / mimo-2.6)
  - creates `/tmp/model-probe` (when a reboot wiped `/tmp`, every model seat looked dead)
  - waits while another driver holds the global lock
  - rides out outages for up to 48 rounds of 10 min
- Stop it with `~/.config/opencode/scripts/autoresearch-loop.sh stop`. Log: `.autoresearch-loop.log`.
- Worker model list: `.autoresearch-models.env`, set by the user; mimo-2.6-flash is in it.
- The driver prepends `.autoresearch-directive-aegis.md` to every worker prompt, via
  `AUTORESEARCH_DIRECTIVE` in the launcher.
- A separate project, `../fdv`, also runs this driver. There is a machine-wide lock, so only one project
  loops at a time.

## 7. Director duties (things the workers can't be trusted with)
- **Audit every KEEP.** Open the cited file, check that the header's `pose_noise_cfg` and flags match
  the claim, and grep the rig diff for teleports or arithmetic edits to coverage. Mark fakes
  `status: invalid`.
- **Watch `autoresearch_research.jsonl` for invalid JSON.** Workers write bad lines; I repaired 10
  (the raw text is kept). The directive now requires validated appends.
- `benchmarks/restroom_sim.py` is audited but is a **synthetic proxy only**. It has a `--selftest` and
  every row carries `metric_class: synthetic_proxy`. Its jerk gate never fires, and its old
  SCRIPTED_BASELINE of 0.8125 was measured on corrupt runs, so don't use it for decisions.
- Keep disk free space above 10 GB (currently 13 GB). Clean with
  `rm -rf /tmp/pip-* ~/.cache/pip ~/.cache/huggingface/hub/tmp*`.

## 8. Still needed from the user
- Optional: API keys for Groq / Gemini / Mistral, to take literature-scan load off the free pool.
  Checklist: `/media/pope/projecteo/connect/connect_gpu/llm.env.example`, router `llm_route.sh`.
  Store keys in `~/.llm.env` (the NTFS drive ignores chmod).

## 9. Paper story (current)
1. Honest physical benchmark (rig v2) and the measurement pitfalls we found.
2. Trochoidal scrub path: B 13/20 → 20/20 (p=0.008) at no learning cost.
3. Pose error is the dominant failure mode (I5 curve), which motivates depth-registration Tier-2 (I9).
4. Hard, conformally calibrated Tier-4 gate (I2/I7).
5. Edge VLA: frozen SmolVLM plus a one-step expert (I4), at ≤25 ms.

Cite only verified sources: π0.7 (arXiv:2604.15483), TinyVLA 2409.12514, SteerVLA 2602.13193,
QVLA (ICLR'26), RoboTTT 2607.15275, Shortcut 2410.12557, MeanFlow 2505.13447, Consistency Policy
2405.07503, RPL 1812.06298, Residual RL 1812.03201, EquiBot 2407.01479, EquiDiff 2407.01812,
Conformal 2107.07511.
