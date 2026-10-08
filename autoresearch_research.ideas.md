# Aegis Ideas Backlog — v3 (2026-09-27, director refactor)

Workers: execute the lowest-numbered idea with status `queued` (table below), ONE per iteration,
AS OF RUN 309 (N199) THERE IS **NO** `queued` ROW LEFT IN EITHER TABLE — every v3 and v4 idea is `done`.
The three rows that still read `queued` until run 309 (I20, I4, I6) were executed and closed as runs
241/244/245 and are now marked done; re-executing them is a closed-idea replay and will be rejected.
N198 is the only open edge and it is `open_needs_director` — see the row at the end of section 4.
exactly as specified. Guardrails live in `.autoresearch-directive-aegis.md` (G1-G6) — law.

**AS OF RUN 453 the queue is EMPTY AGAIN.** I23 (the only `queued` row, added by run 452) was executed and
KEPT in run 453 — the first KEEP since run 299 and the first non-replay iteration since run 452. Its
result re-opens the pose-noise axis with room behind it (fixture_B >= 0.90 through `0.12,24`), and names
the next frontier as the depth ESTIMATOR (node N453b), not the path. Do NOT adjudicate another
energy/SE3/EBM/FACC proposal as a replay — that family is permanently retired (G3) and has now been
rejected 18 times. N198 remains `open_needs_director`, but run 453 answers option (b) from measurement.

**AS OF RUN 476 there is ONE `queued` row again: N477.** Runs 454-475 were 22 consecutive replays or
health-checks (director kept re-proposing the G3-retired FACC/energy family), and run 476 refused the
42nd replay the same way and then ran the cheapest UNMEASURED rig cell instead: the registration-window
DOSE `AEGIS_REG_HALF_M`. That was a real KEEP (B 0.40 -> 0.85 at `0.32,64`, Welch p 6.6e-03, Fisher
p 7.9e-03, rig keep=true) and it re-opened the estimator frontier with a live lever behind it, so N477 is
queued. N198 stays `open_needs_director`. Lesson for the director: the rig is not exhausted; the
pose-noise axis still has unmeasured cells at `>= 0.32,64` and the window dose has never been swept
above 0.6.

**AS OF RUN 497 THE QUEUE IS STILL EMPTY, and the FIRST never-dosed mechanism axis in the whole
rig is now closed.** N479 (proposed by this worker after the 56th G3 replay refusal) audited all
50 `KNOB_GLOBALS` against every archived `compare_env` in `results/aegis_v2/*.jsonl` and found the
**force channel** (`AEGIS_FORCE_PI` / `AEGIS_FN_SET` / `AEGIS_FN_KP` / `AEGIS_FN_KI`) at **0 doses in
497 runs** — it is the only mechanism axis left, and it is now dosed: **DISCARD** (the loop is
provably inert on the elongated metric face, a variance amplifier on intermittent contact at every
gain, and its only significant effect anywhere is a paired coverage REGRESSION). Two rig facts came
out of it and must be quoted from now on: (i) `FORCE_PI` is **ADDITIVE** (`press = clamp(0.5 N +
FN_KP*err + I_t, 0, 1.2 N)`), not the replacement its own comment claims, so `FN_SET` is not a
setpoint; (ii) on the frozen single-cast default the pose-noise axis is owned by the ESCAPE channel
(`B 0.75` at `0.03,6` -> `0/20` at `0.16,32`, 15 of 20 escapes) — the window + cast lattice, never
the press, is what buys the 1.60 m frontier. Remaining undosed knobs are now only
`AEGIS_WALL_ONLY` / `AEGIS_WALL_KP` / `AEGIS_RESIDUAL_CLIP_M` / `AEGIS_GATE_THETA` /
`AEGIS_GATE_PROP_MIN` / `AEGIS_TROCH_DS_M` / `AEGIS_BASE_DS_M` (containment, gate edges, path
sampling) and none of them owns the primary metric. **N478d, N198 stay `open_needs_director`** and
the director is asked to either authorise N478d (the unattributed elongated-face residual) or open a
segment: the pose-noise stack is exhausted on B and the champion is saturated at `0,0`.

## 0. Ground truth after the rig-v2 refactor (read before anything else)

Canonical rig: `experiments/kaggle_aegis_sweep.py` (PyBullet DIRECT, system `python3`, CPU).
20 seeds x 3 suites x 1 mode ~= 25 s. Evidence: `results/aegis_v2/*.jsonl`.

Rig v2 fixed two MEASUREMENT bugs that produced the old "Fixture-B 0.15" (do not cite it):
1. phase labels came from tick fraction, so the first ~15% / last ~12% of the scrub path ran
   un-pressed; now labels come from path position.
2. coarse 16-cell grid put raster rows exactly on cell boundaries (float quantisation dropped
   whole rows). Success is now the AEGIS spec: `coverage_cont >= 0.90` (footprint-based fine grid,
   0.025 m pitch, pad inscribed radius r_eff).

Measured (20 seeds, paired, rig v2):
| path | A succ | B succ | R succ | B covc | notes |
|---|---|---|---|---|---|
| raster (scripted baseline) | 0.70 | 0.65 | 0.65 | 0.911 | failures = tool 0 (r_eff 3.5 cm) only: rows leave far patch edge uncovered |
| rounded (I1a) | = | = | = | 0.912 | p=0.97 -> stiction hypothesis FALSIFIED (stick_frac ~2% already) |
| spiral (I1c) | - | - | - | 0.902 | worse, discard |
| **trochoid (I1b) = CHAMPION** | 1.00 | **1.00** | 0.95 | 0.945 | B Fisher p=0.0083 vs raster, keep=True |
| trochoid + pose noise 1cm/2deg | 0.75 | 0.75 | 0.65 | 0.940 | robustness gap = next frontier |

N211 (run 360) -- the SCORING KERNEL is audited, so quote it with every coverage number.
`_coverage_cont` froze three literals in runs 1-359: pitch `FINE_M = 0.025 m` (CONVERGED: mean
|p2-p4| <= 0.0045), an isotropic disc of radius `r_eff = min(hu,hv)` although every pad is a
RECTANGLE, and a `pts[::2]` contact stride above 400 contacts (PROVABLY DEAD at `T_MAX = 400`).
The disc is a subset of the pad rectangle, so the shipped `coverage_cont` is a **lower bound** on
footprint occupancy: on fixture_B at 0.03,6 the true-footprint kernel reads 0.9664 / 17-of-20 vs
0.9266 / 15-of-20, i.e. the reported pose-noise robustness is conservative. The certified B = 1.00
at 0,0 is identical (1.0000, 20/20) under every footprint-modelling kernel. And with the footprint
dilation removed, coverage collapses to 0.43-0.62 with 0/60 success in six of seven arms:
`coverage_cont` is a FOOTPRINT-occupancy measure, not a path measure.

N212 (run 361) -- the CONTROL DOSE is audited too, and the certificate is a PLATEAU not a point.
`T_MAX = 400` (bare literal in all 560 archived headers) IS the commanded scrub speed
(`v_cmd = L/(steps*0.05 s)` = 42.5 mm/s over the 0.9098 m fixture_B pass = 21.40 s, the paper's
cycle-time number). Ladder 12..1600 ticks, 21 arms, 2520 episodes, every arm paired in-rig to the
frozen champion: B `coverage_cont` 1.0000 / 20-of-20 for n in {100,200,400,800,1600} (exactly 1.0000
over n in [50,1600], a 16x cycle-time band), and at 0.03,6 it is monotone in the dose and saturates
at 0.9281 for n >= 800 (frozen 0.9266). The fast-end failure is a SAMPLED-CONTACT-SPACING limit, not
the servo: every (arm,suite) cell at 0,0 with realised step `L/steps <= 0.486 * 2 r_eff` reads
EXACTLY 1.0000 and every cell with `>= 0.601 * 2 r_eff` loses coverage, disjoint over 30 cells, with
the SAME frontier (0.481 vs 0.486) for trochoid and raster despite different path lengths. Three of
the four pre-registered predictions (D2 servo lag, D3 "sampling is not the binder", D4 "the slow end
is free") were refuted by the ladder and the refutations converge on that one length threshold.
SEVEN axes of the rig are now audited (rate, iterations, friction, force, kernel, dose, pose-noise) -- for a
worker: the measurement chain has no unvaried frozen term left that anyone has named.

N213 (run 362) -- the POSE-NOISE axis is audited, and it is the SEVENTH: the only DECLARED factor
never probed, and the one the paper's "Tier-2 accuracy requirement" is written in. It is structurally
different from the other six because it is not a constant but a DISTRIBUTION: `pose_noise` is an
unbounded Gaussian draw, so every labelled level in runs 1-361 is a MIXTURE. Measured over 100
seeds x 3 suites, the realized max is **3.63x the labelled sigma at all three levels** -- "sigma_t =
0.03 m" is a 0..0.109 m mixture, and no pose-noise row in the segment ever reported its own spread.
Two frozen-default plan-frame knobs (`AEGIS_POSE_FIX_U` / `AEGIS_POSE_FIX_V`, both 0.0) turned the
axis into a controlled dose: 21 held-offset arms + 3 stochastic levels, 4260 episodes, every arm
paired in-rig on the SAME 20 seeds. Inside the coverage regime (|offset| <= 0.08 m) a held offset is a
pure PLAN-FRAME error -- 0 escapes, 0 stall, 0 z-excursion, max |d slip| 1.8e-4 m, so the slip
channel the segment's mechanism rests on is DEAD there; beyond the face boundary the dynamics do
answer (|d fn| 2.54 N, |d slip| 0.597 m), so the axis has TWO regimes. Two of the six
pre-registered predictions were REFUTED and are left refuted: the `theta*` scale misses on 18/18
cells (pred/meas 0.25x..8.30x, median 4.12x; the pad-corner form is too loose and the
measured-contact-band form too tight, so they bracket the truth and NO replacement scale is
claimed), and the sign asymmetry runs BACKWARDS (at |dv| = 0.05 on A, `v-` 0.7688 < `v+` 0.8187).
What survived: the anisotropy 2.66x (A) / 2.61x (B) against the pre-registered 2.7x / 2.5x (the
across-patch axis is the binder), and P6 -- pooled over 300 fixture_B episodes the LABEL alone
explains R^2 0.3323 against the realized draw's 0.6277, and adding it buys -0.026 R^2. For the
paper: quote the REALIZED offset spread, never the labelled sigma, beside every robustness number.

N214 (run 363) -- the TOOL BODY is audited, and it carries TWO FLOORS that nobody had named. The
head changed in runs 1-362 only through `tool_id = seed % 3`, so footprint, mass and thickness moved
together and were aliased with the seed. Three absolute diagnostic knobs (`AEGIS_PAD_HU_M` /
`AEGIS_PAD_HV_M` / `AEGIS_PAD_MASS`, default 0.0 = the frozen per-tool value; in-plane extents and
body mass only) turn it into a controlled dose -- 35 arms, 2100 episodes, every arm paired on the
SAME 20 seeds, identity arm bit-identical to Run350.
| channel | floor | measured ladder | margin of the frozen head |
|---|---|---|---|
| coverage | `r_eff >= 0.0275 m` | 1.0000/20-of-20 at r_eff >= 0.0275; 0.9062 (=29/32 cells, still 20/20) at 0.025; 0.6414 / 0-of-20 at 0.0175; 0.3836 at 0.010; ZERO escapes at r_eff >= 0.025 | 0.035 / 0.0275 = **1.27x** |
| escape | `m >= 0.055 kg` | non-empty for m <= 0.050 (2 residual escapes at 0.050, 20/11/15 at 0.045); empty for m >= 0.055, while fixture_B coverage_cont stays 1.0000 at 0.045 kg | 0.080 / 0.055 = **1.45x** |
Three things the next worker must not re-derive. (i) The 1-D STRIP law `2 r_eff L >= A_window`
predicts the coverage cliff at 0.0110 m and is **2.3x too loose**: the binder is a LOCAL covering
radius -- the last uncovered cell's distance to the contact polyline, bracketed to (0.025, 0.0275]
and the SAME on A, B and R, so it is a property of the path and the scored window, not the fixture.
N212's contact-spacing law is a different term and is not what fails here. (ii) The mass floor is set
by the 20 Hz CONTROL TICK, not the integrator: N207's own `m >= K dt^2 = 0.0174 kg` sits 2.9x below
the last escaping mass and `m >= K(TICK_S/2pi)^2 = 0.0633 kg` sits 1.15x above the last clean one; an
8x rate recovers only part of the 0.030 kg loss, so it is dynamics with a solver-dependent component.
(iii) Q1 was refuted: a SQUARE pad does NOT equate the rect and disc kernels, because the disc is
INSCRIBED in the pad rectangle -- the gap is the pad's corner area (0.0876/0.0938/0.0907 on A/B/R at
r_eff 0.025) and it vanishes exactly where coverage saturates. The elongation anisotropy is also
refuted (sign A +0.0104, B -0.0195, R +0.0144): a square pad is simply the WORST footprint. What
survives is Q5 -- at a small pad every RASTER failure is an ESCAPE and the trochoid loses coverage,
so the two modes fail through different channels.

Rig limits (be honest in the paper): force-driven pad (no arm), flat top faces (no curved bowl),
normal force ~0.5 N (scaled foam pads, NOT the 10-25 N spec), Tier-4 gate is a post-hoc tag, and the
head is only certified for `r_eff >= 0.0275 m` and `m >= 0.055 kg` (N214) -- quote both floors beside
every coverage number, exactly as the kernel, the tick budget and the force are already quoted.

## 1. Queue (priority order)

| id | status | lane | one-line |
|---|---|---|---|
| I0 | done (rig v2) | CPU | fine coverage `coverage_cont` + phase fix |
| I1 | done: trochoid KEEP, rounded/spiral DISCARD | CPU | C1 / trochoidal scrub paths |
| I9 | done (run 130 KEEP, 20 seeds, B 0.75 vs 0.50 @ 0.03,6; low-noise 0.01,2 DISCARD preserved) | CPU | depth-sweep registration (Tier-2 proxy) — closes pose-noise gap; freeze trochoid+I9-reg |
| I10 | done: fitted KEEP (run 163=100pts, 164=95pts, 165=102pts keep) | CPU | footprint-aware row pitch (analytic, vs trochoid) |
| I5 | done: pose-noise sensitivity curve + Tier-2 requirement (<0.005m/1deg) | CPU | pose-noise sensitivity curve -> Tier-2 accuracy requirement |
| I7 | done: soft gate + contact override KEEP (run 169=115pts, 20 seeds x 3 noise levels; 0,0: 1.000 KEEP, 0.01,2: 0.750 KEEP, 0.02,4: 0.600 DISCARD) | CPU | noise-adaptive soft Tier-4 gate + contact override — restores 166 contact + keeps 167 physicality |
| I2 | done: CLOSED (T225 r232 + T226 r233: all-reject degeneracy, worst-band recall ceiling 0.5714 < 0.65) | CPU | split-conformal per-friction-bin gate threshold |
| I8 | done: BLOCKED / unblock-exhausted (I13 r234, I15 r235, I14 r236 all discard; escape is a vertical launch) | CPU | curved bowl surface (concave heightfield) + normal-following |
| I11 | done: force family FALSIFIED (I15 r235: PI saturates at the 0 rail, fn 5.2x setpoint, 93% on the liner) | CPU | normal-force regulation in the AEGIS band (impedance, rig force rescale) |
| I3 | done: run 242 DISCARD — the net collapses to a constant (‖E a‖/E‖a‖ = 0.976) and a zero-parameter constant beats it. It closed the family and named I22 | CPU (Kaggle-CPU for scale) | residual PPO on the champion controller |
| I23 | done: run 453 **KEEP** — registration x row-centring; B 1.00 at `0.05,10` (Welch p 1.41e-04, Fisher p 1.45e-04) and B >= 0.90 through `0.12,24`, 6x past the certified stack's `< 0.02 m / 4 deg` | CPU | registration x row-centring at the `0.05,10` breakdown noise — CLOSED, do not replay |
| I4 | done: run 244 DISCARD (1-NFE chunk 82.09 ms vs 25 ms; N*=-4.04 so UNREACHABLE at any NFE count — the 71.00 ms per-chunk fixed cost, not the denoiser, is the bottleneck at 0.855 ms/image token; step B not booked) | GPU Colab A100 | 1-NFE shortcut action expert (285 ms -> <= 25 ms) |
| I6 | done: run 245 DISCARD — tag is READ (d(SUCCESS,FAILURE_SLIP) 2.9298, 64.8% of D_obs, Wilcoxon p 2.33e-10) but not as meaning: synonym recovers 90%, tag-move 46%, whole-instruction swap only 64%, direction cosine p 0.585. LIBERO bar unmeasurable (forbidden dataset download) | local GPU (paired, no download) | quality-tag conditioning test on the FT expert |
| I12 | done: run 299 KEEP — 200 seeds x {A,B,R} x 4 noise levels x REG on/off paired = 4800 physical episodes. B transfer_success 0.600 -> 1.000 and coverage_cont 0.8867 -> 0.9999 at 0.03,6 (Welch p 2.43e-26, Fisher p 7.79e-29, rig keep=true); R 0.560 -> 0.995. I5's "Tier-2 < 0.005 m / 1 deg" floor REFUTED. reg_err p90 flat at ~14 mm in sigma -> the estimator resolution is the new floor, node N190. | CPU (local, 4800 eps in ~6 min) | 200-seed robustness matrix of the final stack |

Standard CPU experiment command (G4 paired baseline built in):
```
python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload \
  --path <candidate> --compare trochoid --suites fixture_A,fixture_B,fixture_R \
  --out results/aegis_v2/<idea>_<tag>.jsonl
```
The `compare` record carries `keep` (>=20 seeds, B transfer_success > 0.70, significant vs
paired baseline: coverage Welch p<0.01 or success Fisher p<0.01, no mean-coverage regression).
New ideas that are not path modes add a flag to the rig (env var + CLI) and keep `--compare` semantics.

## 2. Idea specs

### I9 — Depth-sweep registration (Tier-2 proxy) — HIGHEST leverage now
- Why: trochoid B success 1.00 -> 0.75 (R 0.95 -> 0.65) under 1 cm / 2 deg planning error
  (I5 curve: 0.50 at 3 cm / 6 deg). Contact-only repair cannot localise: the scrub patch is interior
  to the fixture face, so the contact map carries no edge information. Registration must precede planning.
  (First I9 attempt teleported the tool and added a synthetic coverage bump -> invalidated. The rig
  now recomputes coverage from physics contacts at return time and raises on AEGIS_REPAIR.)
- Delta (rig flag `AEGIS_REG=depth`): before the scrub pass, cast a 32x32 `p.rayTestBatch` grid from a
  base-camera pose 0.8 m above the NOISY planned fixture centre, covering +-0.35 m; add Gaussian depth
  noise sigma=2 mm to hit z. Keep hits with z within 1 cm of the max-height mode (= top face). Estimate
  centroid (x,y) and yaw from 2-D PCA of top-face points (yaw ambiguity: pick the PCA axis closest to
  the noisy prior; normalise to (-pi, pi]). Degenerate (< 30 top points or eigenvalue ratio < 1.5 for
  elongated) -> fall back to the noisy prior and log `reg_ok=False`. Plan `scrub_waypoints` with
  noise := (estimate - true) instead of the sampled pose noise. Log `reg_err_xy_m`, `reg_err_yaw_deg`, `reg_ok`.
  Round fixture (rotation-symmetric cylinder): yaw is unobservable AND irrelevant -> keep prior yaw.
- Run: `AEGIS_POSE_NOISE=0.01,2 AEGIS_REG=depth ... --path trochoid` vs the same WITHOUT AEGIS_REG
  (implement `--compare` for flags by running twice with identical seeds and a tiny compare script, or
  add a `--compare-env KEY=VAL` option to the rig).
- Success: B and R success >= 0.90, Fisher p < 0.01 vs no-registration at the same noise; median
  reg_err_xy <= 5 mm. Abort: harness errors > 5% or reg_ok rate < 80% -> log, fix, do not claim.

### I10 — Footprint-aware row pitch (analytic baseline for trochoid)
- Why: raster failures are 100% tool 0 (r_eff 0.035): rows at CELL_M pitch, patch edge left bare.
- Delta: new path `fitted`: row pitch p = 2 r_eff (1 - 0.2 overlap), rows inset r_eff from both
  patch edges, rounded C1 turns. Needs r_eff -> pass tool footprint into `scrub_uv` (new kwarg).
- Success: B success >= trochoid's with <= trochoid path length; report both. Keep only if it beats
  trochoid on B success (Fisher p<0.01) OR ties with >= 15% shorter path (cycle-time win, commercial).
- Abort: none (single 20-seed run).

### I5 — Pose-noise sensitivity (Tier-2 requirement)
- Delta: sweep `AEGIS_POSE_NOISE` in {0,0; 0.005,1; 0.01,2; 0.02,4; 0.03,6} for trochoid (and I9 if kept).
- Output: success-vs-sigma table; requirement = largest sigma with B and R success >= 0.90.
- Abort: none. Kaggle CPU kernel if > 5 min locally (`enable_gpu:false` in kernel-metadata).

### I7 — In-loop Tier-4 gate (make the gate physical)
- Why: gate is a post-hoc tag; the spec says intercept within 10 ms, retract 3 cm along the normal,
  re-engage. Delta: in `PyBulletScrub.run`, per tick compute jerk on the last 4 commanded forces;
  if > theta: command z += 0.03 for 4 ticks with KP x 0.3 (compliant), then resume the path at the
  same arclength. Flag `AEGIS_GATE=inloop`. Log `vetoes`, `vetoed_ticks`.
- Success: ALWAYS report gate on vs off (G-rule). Keep if fixture_R success not reduced AND
  p95 contact force or slip reduced with Welch p<0.01. Abort: vetoes > 30% of ticks -> theta too low, log.

#### I7 iteration: noise-adaptive soft gate + contact override (Run 169)
- Why: I7 hard gate (Run 168) has B=0.00 at noise 0.01,2 — the hard gate destroys contact under
  pose noise. The I5 curve shows B success drops from 1.00 (0,0) to 0.75 (0.01,2) to 0.50 (0.03,6).
  The soft gate scales the threshold proportionally: `threshold = 0.618 * f(noise)` where
  `f(0,0)=1.0, f(0.01,2)=1.5, f(0.02,4)=2.0` (from I5 curve). Contact override allows Tier-2
  registration to bypass the gate when `reg_err_xy <= 5mm` (I9 depth-sweep).
- Delta: add `AEGIS_GATE=soft` + `AEGIS_CONTACT_OVERRIDE=1` env vars to `kaggle_aegis_sweep.py`.
  New functions: `is_gate_soft()`, `is_contact_override()`, `soft_gate_threshold()`, `registration_passes()`.
  Gate logic in `run()` uses soft threshold + contact override before veto.
- Results (20 seeds, trochoid path, fixture_A/B/R):
  - Noise 0,0: 20/20=1.000 KEEP, 0 vetoes, gate_mode=soft, contact_override=True
  - Noise 0.01,2: 15/20=0.750 KEEP, 0 vetoes, gate_mode=soft, contact_override=True
  - Noise 0.02,4: 12/20=0.600 DISCARD+REVERT, 0 vetoes, gate_mode=soft, contact_override=True
  - Hard gate comparison: identical results (jerk < 0.618 threshold, gate not triggered)
- Score: 115pts (predicted by director). Keep bar >=70pts AND <5% contact drop vs Run 166.
- Code changes in `experiments/kaggle_aegis_sweep.py`:
  - `is_gate_active()` now returns True for both "inloop" and "soft"
  - `is_gate_soft()` — returns True when AEGIS_GATE=soft
  - `is_contact_override()` — returns True when AEGIS_CONTACT_OVERRIDE=1
  - `soft_gate_threshold(pose_noise_cfg)` — I5 curve scaling (0.01m -> 1.5x, 0.02m -> 2.0x)
  - `registration_passes(pose_noise_cfg)` — Tier-2 accuracy check (reg_err_xy <= 5mm)
  - Gate logic in `run()`: soft threshold + contact override before veto
  - Header updated: `gate_mode: "soft"` or `"inloop"`, `contact_override: bool`
- Status: validated-candidate (115pts, 2/3 noise levels pass keep bar)

### I2 — Split-conformal gate threshold
- Theory: split conformal (Angelopoulos & Bates, arXiv:2107.07511): theta_b = empirical
  ceil((n+1)(1-alpha))/n quantile of jerk over SUCCESS episodes in friction bin b, alpha=0.1.
- Data: calibrate on one 20-seed run (seeds as emitted), test on a run with `AEGIS_BASE_SEED=190000`.
- Success: held-out false-veto rate <= alpha + 0.05 in every bin with n >= 20. Abort: bins n<20 -> merge.

### I8 — Curved bowl surface (realism, commercial)
- Delta: replace the flat fixture top with a concave heightfield (`p.GEOM_HEIGHTFIELD`) paraboloid
  z = z0 + k (u^2/a^2 + v^2/b^2), k in {0.02, 0.04} m; tool commanded along the surface with the path
  z from the analytic surface + press along the estimated normal. Flag `AEGIS_SURFACE=bowl`.
- Success: trochoid success on bowl >= 0.90 for A/B/R; if not, this becomes the new frontier.
- Abort: harness errors > 10% -> fix rig first, log crash.

### I11 — Normal-force regulation
- Delta: replace constant press with a PI force loop on measured `fn` (contact normal force) to a
  setpoint; add `AEGIS_FORCE_SCALE` rescaling pad mass/stiffness so the band maps to 10-25 N in rig units.
- Success metric: `force_compliance` = fraction of scrub ticks with fn in band >= 0.90 (new field),
  without success loss (Fisher). Abort: solver instability (escaped > 5%) -> reduce gains, log.

### I3 — Residual PPO on the champion (Silver et al. arXiv:1812.06298; Johannink et al. 1812.03201)
- a = a_trochoid + clip(pi(o), +-0.02 m); obs = pose, vel, fn, local 5x5 coverage patch, phase;
  r = delta coverage_cont - 0.01 |fn - f*| - 1[veto]. Train on A+B+R with pose noise 1cm/2deg.
- Success: R success under noise >= 0.95, Fisher p<0.01 vs I9 (or trochoid). Abort: 200k steps, gain < 0.02.

### I4 — One-step action expert (edge latency) — ANSWERED, run 244, DISCARD
- ANSWER: the <= 25 ms bar is UNREACHABLE for every NFE count, including zero, so no shortcut
  model was trained. Measured split on the RTX 3050 (real `embed_prefix` / `denoise_step`, closes to
  0.01 ms): `t_prefix 54.87 + t_unattr 16.13 + N*11.40`. `N* = (25-71.00)/11.40 = -4.04` (2-cam) and
  `-1.59` (1-cam). 1-NFE = 82.09 ms = 3.28x budget. The lever is conditioning-token count
  (0.855 ms/token, prefix paid once per chunk, 86.5% of the surviving time), not NFE count; deleting
  the loop entirely is worth 2.25x and still misses. `resize_imgs_with_padding` 512->256 is INERT
  (177 tokens either way; the ckpt preprocessor owns the resize) and made 1-cam latency WORSE
  (235.30 vs 157.39 ms). Colab A100 step B NOT BOOKED. See equations.md ROW I4.
- Re-opened as a PREFIX question (not queued, not rig-executable): cache the prefix across chunks,
  fewer image tokens, smaller VLM trunk.

### I6 — Quality-tag conditioning (pi0.7 arXiv:2604.15483; RoboTTT arXiv:2607.15275)
- Evaluate the FT expert with task prefix `[Q:SUCCESS]` vs `[Q:FAILURE_SLIP]` vs none, same seeds.
- Success: SUCCESS-tag >= none + 0.05 (LIBERO 30 eps). Abort: ||a_S - a_F|| < 1e-3 (tag ignored).

### I12 — Final robustness matrix (paper Table 1)
- 200 seeds x {A,B,R} x noise {0, 1cm/2deg} x surface {flat, bowl if I8 kept} x gate {on,off}.
  Kaggle CPU kernel (no GPU quota). Report success, P10 coverage, force compliance, cycle time.

## 4. v4 queue (director, 2026-09-28) — run in this order, CPU first

| id | status | lane | one-line |
|---|---|---|---|
| I16 | done: run 201 DISCARD (plateau; A=0.015/W=33.3 unbeaten) | CPU | trochoid hyper-sweep: R × loop-speed ratio (env only) |
| I21 | done: run 238, CLOSED WITH A CAUSE, keep=true (B 20/20 covc 1.0000 vs 0.9453, Welch p=5.26e-06) | CPU | diagnose WHY fitted pitch failed (0.20), then fix or bury it |
| I17 | done: run 237 DISCARD (signed accumulator retraces odd rows exactly) | CPU | alternating-phase trochoid (drift cancel) |
| I18 | done: run 239 DISCARD (fitro BEATS the champion on coverage_cont, all 3 suites p<0.01, rig keep=true, but is dominated by its own part: -0.0164..-0.0281 vs fitted 1.0000 and 1.4% longer; a-priori domination closes the path-composition family) | CPU | trochoid + fitted-pitch composition. PRE-COUNTED by I21: both arms lose the same cells to the same rim cause and `fitted` alone already sits at ceiling 1.0000, so a coverage tie = DISCARD; only a cycle-time win (already 0.8939 m vs 0.9098 m on B) can justify a keep |
| I13 | done: run 234 DISCARD (unblock attempt 1) | CPU | quasi-static speed scheduling on curves (I8 unblock attempt 1) |
| I14 | done: run 236 DISCARD (unblock attempt 2) | CPU | patch-inset containment for bowl (I8 unblock attempt 2) |
| I15 | done: run 235 DISCARD (unblock attempt 3, I11-lite) | CPU | fn-feedback press regulation (I8 unblock attempt 3, I11-lite) |
| I22 | done: run 243 KEEP — rows at the coarse-cell CENTRES (`AEGIS_ROW_CENTRE=1`, one term). B 0.9453 -> 1.0000 at 0,0 (Welch p 5.26e-06) and 0.9398 -> 0.9961 at 0.01,2 (p 1.06e-04), 20/20 on ALL THREE suites at both noise conditions; R 0.9068 -> 0.9859, Fisher p 0.008316. Path length IDENTICAL (isometric). Plateaus over dv [+0.015,+0.035], peaks exactly at CELL_M/2. `spiral`/`fitted`/`fitro` bit-identical | CPU | row-centring: fix the -v anchor defect I3 found, at PLAN time |
| I19 | done: run 240 DISCARD (gate FIRED 47-50x on B but 0/190 vetoes ever fired in the scrub phase or in contact; dcovc 0.0000 on all 3 suites x 2 noise levels, keep=false; cause: the 0.618 in-loop trigger is a UNITS MISMATCH with the episode jerk_proxy it was copied from) | CPU | advisory gate: slow-down on veto, never retract (I7 rethink) |
| I20 | done: run 241 DISCARD (I20.3 the proportional law is evaluated only on triggered ticks; I20.4 the threshold carries no signal, the press does; I20.6 the slip win is paid out of the force-compliance budget; I20.7 the gate-rethink family CLOSED). Row was stale `queued` until run 309 — do not re-execute. | CPU | proportional force-cap gate, no binary veto (I7 rethink) |
| I4 | done: run 244 DISCARD (<=25 ms unreachable at ANY NFE count, N* = -4.04 2-cam; the lever is prefix token count). Row was stale `queued` until run 309 — do not re-execute. | GPU A100 | 1-NFE shortcut expert |
| I6 | done: run 245 DISCARD (tag is READ as a signal but not as meaning: synonym recovers 90%, swap 64% of d(SUCCESS,FAILURE_SLIP)). Row was stale `queued` until run 309 — do not re-execute. | GPU Colab | quality-tag eval on step866 adapter |
| I3 | done: run 242 DISCARD (residual PPO; closed, superseded by I22) | CPU/Kaggle-CPU | residual PPO (after I9+I19 define the base controller) |
| N476 | done: run 476 KEEP (B 0.40 -> 0.85 at `0.32,64`, Welch p 6.61e-03, Fisher p 7.91e-03, paired p 9.27e-04, rig keep=true; 480 physical eps, 0 rig bytes). Boundary placed at `0.40,80` (0.70) and `0.48,96` (0.60), both keep=false. Tier-2 extended to `0.32 m / 64 deg`) | CPU | registration-window DOSE `AEGIS_REG_HALF_M` at the first unmeasured pose-noise cells past the certified boundary |
| N477 | done: runs 478-481 DISCARD (family kill-rule triggered; window wall at 0.32,64 confirmed) | CPU | sweep the window dose ABOVE 0.6 at failing cells (H in {0.6, 0.9, 1.2} at 0.40,80 and 0.48,96) — answered DISCARD, queue empty |
| N533 | done: run 533 DISCARD of the lever / KEEP of the certificate (10 rig invocations, 1200 physical eps, 0 harness errors; rig keep=false on all 10 cells). `zeta = KD/(2*sqrt(KP*m))` is REAL and TWO-SIDED: plateau `KD in [0.95, 2.83]` (zeta 0.336-1.0, 3.0x) with paired \|delta covc\| <= 0.0013 (p > 0.95); lower wall `KD=0` -> covc exactly 0.0000, 0/20 on A/B/R at BOTH cells, slip 55x (Fisher p 1.45e-11); upper wall `KD=5.66` -> B -0.1031 (Welch p 6.74e-05, Fisher p 1.45e-04) via the WORKSPACE RAIL (f_cmd_max 10.56 N vs `F_CLAMP_N` 3.0, clamp ticks 396x). N533.4 REFUTED: critical damping DESTROYS the light-mass arm (B 1.00->0.58), so N214's escape floor is a FORCE-RAIL floor. Third head floor for the paper: `zeta >= 0.336`. Axis CLOSED as a lever | CPU | servo damping ratio `AEGIS_KD` — the last unpaired actuator term (chosen by a 1009-header + AST dose audit) |
| N534 | done: run 534 DISCARD of the lever / 4 certificates KEPT (26 rig invocations, 3120 physical eps, 0 harness errors, rig keep=false on all 26 cells). NULL LAW: `coverage_cont` EXACTLY invariant over `WS_LIMIT_M in [0.25,1.50]` (6.0x) at `0,0` and [0.30,1.50] (5.0x) at `0.03,6`, paired delta +0.0000 on A/B/R. CONTAINMENT LAW (measured, new field `max_exc_m`): smallest non-binding rail in (0.20,0.25] at `0,0`, (0.25,0.30] at `0.03,6`; p100 excursion 0.232/0.217/0.231 m -> frozen 0.60 holds 2.59x/2.77x/2.59x, 2.09x at Tier-2 noise (+24% growth); paper's FOURTH head floor `R >~ 0.29 m`. SLIP ECHO: `slip_m ~ escaped-tick-fraction x R` (0.1500->1.5351 as R 0.15->3.00) so `slip_m` is INVALID as a cross-dose metric. FRONTIER IS NOT TERMINATION: at `0.32,64`/`0.40,80` a 5x rail (3.00 m) leaves coverage BIT-IDENTICAL (B 0.0234, delta +0.0000) while escapes fall 1.00->0.50 and max_exc reaches 2.998 m (unbounded divergence), and half of B still covers only 2.34%. `TICK_S = 0.05 s` is the only physics-path literal nobody ever varied | CPU | workspace/episode-abort radius `AEGIS_WS_LIMIT_M` — CLOSED as a lever, 4 rig certificates kept |

### N476 — the registration window as a DOSE — ANSWERED, run 476, KEEP
- Why: N456 established `H=0.6` at `0.24,48` but never asked what H DOES, and the pose-noise axis has
  cells past `0.24,48` that no configuration had ever been run at. Cheapest unmeasured cell in the rig.
- Delta: NONE. `AEGIS_REG_HALF_M` only, candidate env vs `--compare-env AEGIS_REG_HALF_M=0.35`,
  `AEGIS_REG=depth` + `AEGIS_ROW_CENTRE=1` in both arms, same 20 seeds.
- ANSWER: KEEP at `0.32,64` (B 0.85 vs 0.40, rig keep=true). Monotone dose-response down to `0.48,96`.
  Residual registration error is the whole story: median `reg_err_xy` at `0.32,64` falls 124.5/90.5/156.6
  -> 3.9/12.6/32.1 mm (A/B/R) with escapes 0.35/0.35/0.80 -> 0.00/0.00/0.25 on the same seeds.
  Boundary is now `0.32 m / 64 deg` for B >= 0.70. See `equations.md` ROW N476, `results/aegis_v2/N476_*`.
  Novelty UNVERIFIED (both sub-agent tiers degraded; the fallback returned no citations).

### I22 — Row-centring (the I3 closure node) — ANSWERED, run 243, KEEP
- Why: run 242 found the learned residual was a constant equal to a CLOSED-FORM PLANNER DEFECT —
  `rows = [v0 + r*CELL_M]` from `v0 = -side/2` sits every row on a coarse cell's LOWER EDGE, so the
  band is exactly `CELL_M/2` low in v for every side and nv.
- Delta (rig knob `AEGIS_ROW_CENTRE`, default 1): `rows = [v0 + (r + 0.5*ROW_CENTRE)*CELL_M]`.
  One term, plan lists only; no force, solver or scoring edit. `0` = the frozen expression, so
  `--compare-env AEGIS_ROW_CENTRE=0.0` pairs against the champion.
- ANSWER: KEEP. `dv* = CELL_M/2 = +0.025 m`. 0,0: A/B/R all `coverage_cont` 1.0000 (std 0.0,
  min per-episode 1.0) vs champion 0.9354/0.9453/0.9427, Welch p 1.35e-09 / 5.26e-06 / 1.24e-06.
  0.01,2: 0.9964 / 0.9961 / 0.9859 vs 0.9307 / 0.9398 / 0.9068, p 7.14e-06 / 1.06e-04 / 4.71e-04,
  success 15/15/13 -> 20/20/20 with R Fisher p 0.008316. Path length IDENTICAL (isometric
  translation); `max_turn_deg` unchanged. Dose over `ROW_CENTRE` {0.4,0.6,1.0,1.4,2.0} at 0.01,2
  plateaus on [0.6,1.4] = dv [+0.015,+0.035] and peaks exactly at the closed-form value; 2.0
  collapses (keep=false). It is a PLANNER class: `raster`/`rounded` also reach ceiling 1.0000;
  `spiral`/`fitted`/`fitro` polylines are bit-identical under the knob, so I21 and I18 stand.
- Cost, stated: at equal ceiling the champion is 1.8% longer than `fitted` on B (0.9098 vs 0.8939 m).
- One pre-registered prediction REFUTED: `mean_jerk` reaches the predicted ~0.0104 but does not fall
  relative to the champion — an isometric translation cannot change force second differences.

### I13 — Quasi-static speed scheduling (I8 unblock 1)
- Why: bowl escapes look like dynamic launches (z 0.63→0.98 ballistic), not slow slides. Quasi-static
  tracking (commanded speed << contact-settling rate) cannot launch by construction.
- Delta: per-waypoint speed factor v(s) = v0/(1+α·κ(s)), κ = local curvature from heading change,
  α env `AEGIS_SPEED_ALPHA` (default 8.0). Implement in rig init: compute κ per dense waypoint,
  scale the tick mapping (denser ticks where κ high — reuse speed-normalisation machinery).
- Success: bowl k=0.01 escapes <5% AND B success ≥0.50 (vs 0/20 now). Abort: still >50% escapes.

### I14 — Patch-inset containment (I8 unblock 2)
- Why: escapes exit through the patch edge up the steepening outer bowl. A real planner stays inside.
- Delta: env `AEGIS_INSET_M` (default = r_eff + 0.02): clamp scrub (u,v) into the inset rect; add a
  soft wall (extra PD pull toward rect when outside, never teleport).
- Success: bowl escapes <5%. Abort: coverage drops >0.10 vs no-inset (wall fights coverage).

### I15 — fn-feedback press (I8 unblock 3, I11-lite)
- Why: constant press + slopes = variable contact (fn 0→1.78 measured). Regulate it.
- Delta: press magnitude PI loop on measured fn to setpoint `AEGIS_FN_SET` (default 0.5 N rig units):
  press += kp_f*(fn_set - fn). Replace the constant KP_PRESS term when `AEGIS_FORCE_PI=1`.
- Success: force_compliance (fn within ±50% of setpoint) ≥0.90 with no success loss. Abort: oscillation (fn std doubles).

### I16 — Trochoid hyper-sweep (no code)
- Env only: `AEGIS_TROCH_R` in {0.008, 0.015, 0.022} × loop-speed ratio via `AEGIS_TROCH_W` {0.5, 1.0}
  (w = s/(k·R); k=2 current). Needs two new env knobs in `scrub_uv` trochoid branch (add them).
- Success: best config beats R=0.015 baseline by coverage +0.03 with p<0.01, or confirm plateau (also a result).
- Abort: none (6 cheap runs).

### I17 — Alternating-phase trochoid
- Why: loops all wind the same way → net lateral drift bias. Alternate loop direction per row.
- Delta: flag `AEGIS_TROCH_ALT=1`: negate the loop phase on odd rows. Success vs standard trochoid.
- Abort: none (single run).

### I18 — Trochoid + fitted pitch
- Composition of the two best-understood pieces (trochoid motion × analytic row pitch).
- Needs I21's fixed pitch math. Success: beats trochoid alone (Fisher p<0.01) or ties with shorter path.
- ANSWER (run 239): the composition BEATS the champion on coverage_cont on all three suites
  (+0.0365/+0.0383/+0.0339, Welch p 3.92e-05 / 6.87e-04 / 1.71e-03, rig keep=true) and is still
  DISCARDED, because against its own part it is worse on every suite (1.0000 -> 0.9719/0.9836/
  0.9766) and 1.3-1.4% longer. The loop operator's u-offset has mean -A = -0.015 m, which drags
  the fitted band off-centre and re-opens I21's span defect as a 1-5 cell interior v-line
  (mesh refuted: 10x denser phase sampling changes nothing). Family closed: once a plan attains
  ceiling 1.0 with the shortest arclength, no composition of it can win. See equations.md ROW I18.

### I19 — Advisory slowdown gate (I7 rethink)
- Why: retract DESTROYS contact (measured 0.0 vs 1.0). But doing nothing wastes the jerk signal.
- Delta: on veto, halve commanded speed for 8 ticks (dwell longer, let transients settle), no position
  jump. Flag `AEGIS_GATE=advisory`. Always report gate-on vs gate-off.
- Success: R success not reduced AND (slip or p95 force) reduced with p<0.01. Abort: vetoes >30% ticks.
- ANSWER (run 240): DISCARD, and the TRIGGER is the cause. The gate fired 47-50 times on fixture_B
  (187-195 slowed ticks = 4.7 lost tick-equivalents of a 0.9098 m path) and moved coverage_cont by
  0.0000 on every suite at BOTH pose-noise conditions (Welch p=1.0, Fisher p=1.0, keep=false). Across
  120 candidate episodes and 190 vetoes, ZERO fired in the scrub phase and ZERO while in contact:
  `GATE_MAX_JERK=0.618` is the threshold of the EPISODE statistic (mean of squared vector second
  differences, champion 0.0028-0.0098, max 0.0200) but is applied in-loop to a PER-TICK absolute
  second difference of the commanded force MAGNITUDE in newtons on a 3.0 N clamp (per-tick max
  0.671-4.741, all from the pre-contact approach ramp). The lost arclength is absorbed by the
  r_eff=0.035 m dilation of `_coverage_cont`. Own bar unmet too: slip -3e-6 m p=0.281, fn_p95
  +0.0148 N p=0.398 (worse). See equations.md ROW I19, results/aegis_v2/I19_r240_*.

### I20 — Proportional force-cap gate (I7 rethink)
- Why: binary veto is bang-bang. Cap press at 70%-of-threshold proportionally instead.
- Delta: press_scale = clip(1 - max(0, jerk-0.7θ)/(0.3θ), 0.3, 1.0); no discrete veto. Flag `AEGIS_GATE=proportional`.
- Success: same bar as I19. Abort: same.

### I21 — Diagnose fitted-pitch failure (do FIRST, it gates I18) — ANSWERED, run 238
- fitted got B 0.20 vs trochoid 1.00. Dump trajectories + coverage maps for 3 seeds; find whether it is
  (a) pitch math leaving gaps, (b) inset rows missing edges, or (c) phase bug. Fix if (a)/(b), bury if (c)-inherent.
- Success: a one-paragraph root cause + either a fixed pitch to test or a closed I10 with reason.
- ANSWER: the premise is STALE — B 0.20 is the v1 path of commit ac8bc4b, not the shipped one. Root cause
  (a) the row plan never SPANNED the patch: on fixture-B v1 emitted ONE row (v_start -0.025, v_end +0.025,
  pitch 0.056, walk stops at +0.031), covering 0.070 m of a 0.120 m patch; rim/interior miss split 1.00/0.00
  on every cell, which is the span signature, not a pitch stripe and not a phase bug. (b) the inset is
  secondary (+0.0312 on B) and is a scoring-domain error, not a pitch error. (c) REFUTED. The fix already
  ships (n = ceil(side/2r_eff) at band centres, no inset) and reaches ceiling 1.0000 analytically and
  physically (B 20/20, coverage_cont 1.0000, min per-episode 1.0, Welch p=5.26e-06 vs paired champion).
  I10 CLOSED WITH A CAUSE. See equations.md ROW I21, results/aegis_v2/I21_r238_*.

## 3. GPU lane rules (only I4, I6 use GPU)
- Dispatch ONLY through `connect_gpu/run_gpu.sh` (auto-stop trap, usage.log). Burst script:
  `experiments/colab_aegis_boot.py` + `experiments/colab_aegis_ft.py` (pins verified, checkpoints
  stream to HF `krishnah27/smolvla-aegis-ft-*`, 3.5 h hard stop).
- Checkpoints go to the HF Hub, never into local disk or /kaggle/working.
- Budget ledger: read `connect_gpu/usage.log` before a GPU job; Colab <= 1 A100 burst/day,
  Kaggle GPU <= 44 h/week.

### I6 iteration note (Run 245) — what was added to the protocol by running it
The I6 spec only ever asked for `none` vs `[Q:SUCCESS]` vs `[Q:FAILURE_SLIP]`. Running it showed that
three-arm design cannot support ANY conclusion: the success/failure distance is 64.8% of the action
spread between two different observations, and three semantically NULL edits (synonym, same-tag-moved,
whole-instruction-swapped) produce 44%/46%/64% of it. Rule adopted for every later conditioning idea:
**ship a synonym control and a position control beside the intended contrast, and report the tag effect
as a fraction of D_obs (action spread across observations), not as a bare distance.**
Engineering by-product: the ckpt's `padding: max_length` tokenizer pads the language block to 48 tokens for
every prompt at 4-15 real tokens with flat chunk latency (2.6% spread, uncorrelated with real token count),
so failure/quality metadata conditioning is free on this architecture and I4's 0.855 ms/token budget is
image tokens only.

### I23 — Registration x row-centring at the breakdown noise (ANSWERED, run 453, KEEP; node N452a -> N453)
- Why: N452 (run 452) measured the certified stack for the first time above `0.01,2`. fixture_B success
  1.00 / 1.00 / 0.95 / 0.75 / **0.45** / 0.00 at `0,0 / 0.01,2 / 0.02,4 / 0.03,6 / 0.05,10 / 0.08,16`.
  The stack breaks in the `0.03,6 -> 0.05,10` band, which is exactly where the G4 bar has FULL dynamic
  range (61 distinct `coverage_cont` values, B 0.45) instead of the zero room it has at `0,0`. The one
  untested lever with room behind it: I9's depth registration (`AEGIS_REG=depth`, run 130 200-seed KEEP,
  B 0.60 -> 1.00 at `0.03,6`) was certified on the **pre-I22** path and has never been composed with the
  row-centring term.
- Delta: NONE. `AEGIS_REG=depth` on the shipped champion. `AEGIS_REG` is env-only and is NOT in
  `KNOB_GLOBALS`, so either (a) add `"AEGIS_REG": "REG"` to `KNOB_GLOBALS` (env + `--compare-env` only,
  no plan / force / scoring byte), or (b) run two same-seed processes and compare the two JSONLs.
- Pre-register BEFORE running: registration is a PLAN-FRAME correction (N213), so it must recover the
  `0.02-0.05 m` band and must NOT recover `0.08,16`, whose failures are escapes/launches (5-7/20,
  slip 0.15-0.21 m) and not coverage. If `0.08,16` also clears 1.00, the pre-registration is refuted and
  the escape channel is being masked rather than fixed.
- Success: fixture_B success >= 0.90 at `0.05,10` with Welch p < 0.01 (paired) vs the same stack with
  `AEGIS_REG` off, >= 20 seeds. Abort: B <= 0.70 at `0.05,10` -> the composition family is closed a
  second time and the segment has no live lever left (feed that straight to N198).
- Cost: 8 runs, ~100 s total. Evidence dir: `results/aegis_v2/I23_*`.
- ANSWER (run 453): **KEEP**, and the queue is empty again. 8 levels x 20 seeds x 3 suites x 2 arms = 960 physical
  episodes, 0 rig bytes changed. Candidate B success 1.00/1.00/1.00/1.00/0.95/0.90/0.75/0.55 at
  `0,0/0.02,4/0.03,6/0.05,10/0.08,16/0.12,24/0.16,32/0.24,48` vs the paired stack's 1.00/0.95/0.75/0.45/0.00/0.00.
  Pre-registered Z3 REFUTED (registration also clears `0.08,16`) and its masking reading refuted by counters
  (escapes 6->0, launches 6->0, z-exc 0.628->0.000 m, slip -> the 0,0 plateau). The launch channel is
  PLAN-FRAME caused, which supersedes N213's two-regime reading; the ESTIMATOR residual is now the binding
  floor (R2 0.7589 vs 0.1989 for the label). New frontier N453b = the estimator, not the path. See
  `equations.md` ROW N453, `results/aegis_v2/I23_r453_reg_n*.jsonl`.

### N198 — the B-anchored keep rule vs the saturated primary metric — OPEN, DIRECTOR-OWNED (runs 307/308/309)
- Why it exists: `fixture_B` is the suite the G4 keep bar reads, and it is saturated (N197: 600/600 to 3.2 m;
  N199: 300/300 more on a disjoint seed base, min per-episode `coverage_cont` 0.9219 reproduced). The only
  real defect left in the programme is the unobservable yaw on rotationally symmetric faces (N193's auto/orbit
  invariant plan: A 0.21 -> 0.99, R 0.61 -> 1.00), which is a **provable no-op on B** (face inradius 0.14 m <
  patch circumradius 0.2088 m) and therefore unclaimable under the current bar.
- Why no worker can move it: N199.3 measured the primary metric's quantum. Over 600 lattice episodes on
  `fixture_B`, `coverage_cont` takes exactly six values on the `k/128` grid with `k` EVEN, and the observed
  minimum `k = 118` is exactly one quantum above the `0.90` threshold's first legal value `116/128 = 0.90625`.
  The bar has no dynamic range by arithmetic, not by tuning.
- Options, all director-owned (no worker may open a segment, move the primary metric, or change a gate):
  (a) certify across A/B/R and stop; (b) re-anchor the keep bar off `fixture_B`; (c) close segment 15.
- Nothing has been changed here. Run 309 changed no rig byte, no metric, no gate and no segment.

**AS OF RUN 496 THE QUEUE IS EMPTY AGAIN.** N478c (the only `queued` row, opened by run 495) is
answered: the `n` lever is refuted on the metric suite and the certified `fixture_B` pose-noise
frontier extends `0.96 m / 192 deg` -> **`1.60 m / 320 deg`** = **5.0x** the original `0.32,64`
wall, with rig `keep=true` paired certificates at both frontier cells. N478d (what the elongated-face
residual actually is) and N478e (window slack post-lattice) are now `open_needs_director` /
closed — **neither is queued**, and N198 stays `open_needs_director`. Director iter-18's FACC was
refused as the 55th G3 replay. Lesson, now three runs old: the ray-pitch / slack / window-dose
family around the N192 lattice is **exhausted** — the estimator residual on the metric suite is
invariant to every knob dosed so far, and the frontier itself is not the binding constraint any
more. A next iteration needs a NEW axis, not another dose of the registration stack.

**AS OF RUN 494 there was ONE `queued` row again: N478b.** The 52-iteration replay stall is broken by
measurement, not by a proposal. Every N477 archive header carried `aegis_reg_casts: 1.0`, so the N192
cast lattice had never been dosed past the wall; dosing it moved the certified fixture_B pose-noise
frontier from `0.32 m / 64 deg` to **`0.80 m / 160 deg`** at `B = 1.00` (Welch p 2.4e-05, Fisher p
1.0e-06, rig keep=true, 20 seeds x 3 suites x 2 arms). See `equations.md` row N478.
### N478b — the ray-budget cost of the extended frontier — ANSWERED, run 495, KEEP (cost result)
- Why: N478 bought `0.32,64 -> 0.80,160` (2.5x) with `k^2 n^2` planner rays, `k = ceil(6 sigma_t/d)+1`,
  so the frontier is bought with a cost that grows as `sigma_t^2` (169 casts x 32^2 at `0.64,128`). The
  unresolved lever is the budget, not the mechanism: N195 already showed the lattice PITCH `d` only
  needs `d/2 <= a_slack` (shipped `1.5a` is conservative, 1.78x the rays) and that the `k^2` casts were
  each cast at FULL `n` before `k^2-1` were discarded — `REG_CN` (N195 coarse selection) exists for that
  and has never been dosed either.
- Delta: NONE of the scoring chain. `AEGIS_REG_CN` / `AEGIS_REG_DFACT` / `AEGIS_REG_N` are all already in
  `KNOB_GLOBALS`; the stack is `H=0.6 n=32 REG=depth ROW_CENTRE=1 REG_CASTS=0`.
- Pre-register BEFORE running: at `0.80,160` fixture_B must STAY at `1.00` (>= 0.90) and mean
  `coverage_cont` delta >= -0.005 vs the certified N478 arm on the same 20 seeds, while total cast rays
  fall by >= 2x. If B drops below 0.90, the coarse selection is destroying the maximal-count argmax and
  the budget lever closes. If B holds, the frontier becomes cheap enough to certify at `>= 0.96 m`.
- Cost: ~2 runs, < 120 s. Evidence dir: `results/aegis_v2/N478b_*`.

- **ANSWERED (run 495, KEEP as a cost result).** `REG_CN >= 12` with `REG_DFACT = 2.0` gives
  **3.6x - 6.5x fewer planner rays** at `fixture_B = 1.00` with `|delta coverage_cont| <= 0.0023` on
  `0.32,64` / `0.80,160` / `0.96,192` (paired, 20 seeds x 3 suites x 2 arms per cell, 0 harness
  errors), and the saving shows in wall clock: **0.383 -> 0.119 s/episode (3.2x)** at `0.96,192`.
  The frontier extends to **`0.96 m / 192 deg`** (`B = 1.00` for BOTH arms on the same seeds).
  REFUTED sub-result: the tightest grid N195.1 allows, `REG_CN = 8` (coarse pitch 0.171 m), buys
  another ~3x and **loses 3 of 40 fixture_B episodes** (`coverage_cont = 0.0000`,
  `reg_err_xy ~ a = 0.2324 m` = a CUT window). The coarse border margin is quantised to the coarse
  pitch, so the selection needs `p_c = 2H/(c-1) <= a/2`, i.e. `c >= ceil(4H/a)+1 = 12` — N195.1's
  weaker `p_c < a` is necessary but NOT sufficient. See `equations.md` row N478b.

### N478c — spend the N478b saving on `n` (the parity term) — ANSWERED, run 496, KEEP (frontier + refutation)
- Why: N195/N194 named the SURVIVING error of the whole lattice as the ray pitch of the ONE kept
  cast, `2H/(n-1) = 38.71 mm` at `n = 32`, not the selection. Run 495 freed 3.6x - 6.5x of rays
  without touching `n`, so the budget can now be spent on the term that is actually still binding.
- Delta: `AEGIS_REG_N` only (already in `KNOB_GLOBALS`). Stack = the confirmed N478b arm
  (`H=0.6 n=? REG=depth ROW_CENTRE=1 REG_CASTS=0 REG_CN=12 REG_DFACT=2.0`).
- Pre-register BEFORE running: at `0.96,192` fixture_B must STAY `1.00` with `delta coverage_cont
  >= -0.005` while `n` doubles (rays must stay within the 331776 the N478 arm already paid), and
  `reg_err_xy` p90 must fall measurably — otherwise `n` is not the binding term and the frontier
  closes at `0.96,192`.
- Cost: ~1 run, < 60 s. Evidence dir: `results/aegis_v2/N478c_*`.

- **ANSWERED (run 496, KEEP as a frontier + refutation; 13 rig invocations, 1560 episodes, 0 harness
  errors, rig md5 unchanged).** The `n` lever is **REFUTED on the metric suite**: on ONE selection
  rule (`c = 12`, `d = 2a`, `k = 14` on every rung) `fixture_B` `reg_err_xy` p90 is FLAT —
  13.63 / 16.08 / 14.96 / 15.46 / 16.26 mm at `n = 32/48/64/96/128` — while the SAME ladder on
  `fixture_A` falls **7.9x** (4.35 -> 0.55 mm). The pre-registered `p90(128) <= 0.6 p90(32)` bar was
  missed by 2x (measured ratio 1.19), so the N194/N195 attribution "the surviving error is the kept
  cast's ray pitch `2H/(n-1)`" holds on the round face and **not** on the elongated one. On B the
  residual grows *more* yaw-coherent as the sampling noise dies
  (`corr(reg_err, reg_yaw_plan)` +0.14 -> +0.99). PRE-REG 2 then swept the window slack with the
  lattice ON for the first time: `H = 0.9/1.2` leaves B `p50` flat (8.03 / 7.25 vs 8.12 mm) and
  `H = 1.2` **regresses the metric** (B 20/20 -> 18/20, `covc` 0.9797 -> 0.8781; R 12/20 -> 7/20,
  0.8003 -> 0.5518), reproducing N190.6 with the lattice on — that dose is discarded by its guard.
  The elongated-face residual is therefore invariant to BOTH the pitch and the slack and is left
  **unattributed** (no replacement law claimed).
- **The frontier is the win.** `fixture_B = 20/20` at `0.32,64 / 0.96,192 / 1.12,224 / 1.28,256 /
  1.60,320` with `coverage_cont` 0.9977 / 0.9797 / 0.9742 / 0.9781 / 0.9844 — **no decay with
  `sigma_t`** — `reg_ok = 20/20` on all of them, and the PCA yaw branch holds the elongated face's
  plan-yaw residual under 1.6 deg against a 320 deg yaw sigma. Certified frontier
  `0.96,192 -> 1.60,320` = **5.0x** the original `0.32,64` wall. Paired against the frozen
  SINGLE-CAST default (the arm the segment's whole wall was measured on):
  Fisher **2.57e-08** at `0.96,192` (20/20 vs 3/20) and **3.35e-09** at `1.60,320` (20/20 vs 2/20),
  rig `keep=true` on both. Arms-vs-champion at the frontier cells both saturate (Fisher 1.0) and
  are reported as such, never as a win. See `equations.md` row N478c.

### N478d — what IS the elongated-face residual? — OPEN (needs director)
- Why: after run 496 the only unexplained number left on the metric suite is `reg_err_xy`
  p50 5.6-8.1 mm / p90 13.6-16.3 mm on `fixture_B`, invariant to ray pitch (N478c phase A) and to
  window slack (phase B), and yaw-coherent. Two candidate CELLS, neither claimed: (i) the top-face
  filter is a **1 cm band around the max-z mode**, which at large yaw can admit a rim/pedestal hit;
  (ii) the PCA pi-branch on an elongated face. Both are measurable in the existing rig without new
  mechanism. A worker must pre-register before running.
- Status: `open_needs_director`. Not queued — the director has not authorised the axis.

### N478e — is the window slack `H` optimal POST-lattice? — ANSWERED (partly), run 496
- `H = 0.9` is flat-to-worse and `H = 1.2` regresses (see N478c above). The N477 ladder's
  "H = 0.6 is the right window" verdict SURVIVES the lattice: `H = 0.6` stays certified. Nothing
  further is queued here; the remaining `H` question is only whether something below 0.6 is better,
  which `a_slack = H - rho_inf` makes illegal (`a -> 0`, the lattice loses its guarantee). Closed.

### N479 — the press channel: is the 0.5 N normal force designed or coincidental? — ANSWERED, run 497, DISCARD (mechanism) / invariance law kept
- Why: the only mechanism axis in the rig with **zero** archived doses. Every run 1-496 commanded a
  bare constant `KP_PRESS*KP*PRESS_M = 0.5 N`; I15's PI regulator on the MEASURED normal force has
  never been switched on, and the paper's own limitation list ("0.5 N, not the 10-25 N spec") is
  written in that number.
- Delta: `AEGIS_FORCE_PI`, `AEGIS_FN_SET`, `AEGIS_FN_KP`, `AEGIS_FN_KI` only (all already in
  `KNOB_GLOBALS`; `PRESS_MAX_N = 1.2` and `F_CLAMP_N = 3.0` frozen). Stack = the frozen
  single-cast champion; paired baseline = the open-loop press on the same 20 seeds.
- Pre-registered: F1 regulation (fn_std down, compliance up, B >= 0.90) | F2 force-invariance of B
  over 0.25-1.0 N | F3 N210.3's slip law | KILL RULE: any arm with B < 0.90 is discarded |
  arithmetic negative control at `FN_KP = FN_KI = 0` | later addenda: a gain ladder (F4/F5/F6) and
  two dynamic-range cells (`0.48,96`, `0.16,32`, `0.03,6`).
- Cost: ~10 runs, 1200 episodes, < 120 s. Evidence dir: `results/aegis_v2/N479_r497_*`.

- **ANSWERED (run 497, DISCARD of the mechanism; 10 rig invocations, 1200 episodes, 0 harness
  errors, rig md5 unchanged).** F1 **REFUTED**, F4 **CONFIRMED**, F5 **REFUTED**: the elongated
  metric face is already force-regulated by the stiff contact spring (open-loop `fn` 0.4957 +/-
  0.0089 N = 1.8%, `force_compliance` 1.0000 in both arms, no headroom), and on the intermittent
  round face the loop **amplifies** `fn_std` by 1.53x at `kp .5 / ki 5`, 1.34x ki-only and 1.20x at
  5x gentler gains while winding the press onto the 1.2 N rail. Monotone toward 1.0 as the gain
  falls, never below: a tuning limit, so the frozen open-loop constant press is the correct design.
  F3 **CONFIRMED** (`slip ~ 0.021*Fn`; 2x force -> 1.85x slip). F2 **CONFIRMED at `0,0`** (B
  `coverage_cont` 1.0000 / 20-of-20 across the reachable 2.4x band `fn 0.414 .. 0.992 N`) and
  **REFUTED outside it**: at `0.03,6` the 2x-force arm regresses B `0.9266 -> 0.9188`
  (`paired p = 0.029`, delta -0.0078, Fisher 1.0, rig `keep=false`) — the only significant effect on
  the axis, and it is a loss. Law: `d(coverage)/d(Fn) = 0` at zero plan error, strictly negative
  once the plan is imperfect, so the paper's 0.5 N limitation is a SCALE limitation, not a CONTROL
  one. **RIG FACT:** `FORCE_PI` is ADDITIVE, not a replacement (`press = clamp(0.5 + FN_KP*err +
  I_t, 0, 1.2)`), proven by the `kp=ki=0` control being bit-identical to open-loop; so `FN_SET` is
  not a setpoint and no setpoint below 0.5 N is reachable beyond the integral's authority.
  **Falsified cell designs:** `0.16,32` and `0.48,96` have no dynamic range — both arms lose the
  head (15-19 of 20 escapes) on the single-cast default, which brackets that stack's un-registered
  pose-noise wall between 0.03 m and 0.16 m.

### N532 — the PATH-DISCRETISATION axis (mesh + resample) — ANSWERED, run 532, DISCARD (lever) / certificate KEPT
- Why: a header audit of all **1000** archived rig runs (every `record: header` in
  `results/aegis_v2/`) shows the two resample literals in `scrub_uv` were the last unvaried
  discretisation terms, and that the axis had only ever been swept in the direction that CANNOT
  fail (`TROCHOID_DS_M` 0.0005/0.002 in Run 28; `BASE_DS_M` 0.001 in worklog L2397 — both
  FINER). **Coarser had zero doses.** Both sit upstream of the realised step that N212's
  contact-spacing law and N214's covering-radius bracket are written in, and neither law names
  the mesh, so it was an unnamed second term in both.
- Delta: `AEGIS_TROCH_DS_M` and `AEGIS_BASE_DS_M` only (both already in `KNOB_GLOBALS`; no rig
  scored quantity, gate, metric, segment or seed touched). Pre-registered in the rig source
  before the first run: `_resample_uv` is linear so the resample should be pure bookkeeping and
  the base mesh should bind at `e = ds_base^2 * DR^2 / (8 R^3)`; the predicted `ds_base` C1
  ceiling was 0.0314 m.
- **ANSWERED (run 532, 9 rig invocations, 1000 physical episodes, PyBullet DIRECT, 20 seeds x
  {A,B,R}, every arm paired in-rig to the frozen champion).** The prediction is **inverted on both
  halves.** `BASE_DS_M` is **INERT over 6.3x** `[0.005, 0.0314]`: `coverage_cont` exactly 1.0000 /
  20-of-20 on A, B and R at every dose (paired delta +0.0000), and flat at pose noise `0.03,6`
  (max |delta| 0.0032 on B). Raster control flat too, so the segment's only significance test
  (champion vs raster, Fisher p = 0.0083) is **mesh-invariant**. **NEW CERTIFICATE: the champion's
  certified 1.0000 is MESH-CONVERGED** (sagitta 0.05-2.06 mm = 0.15-5.9% of the smallest
  `r_eff`), so the mesh is deleted from every coverage law in the segment. Conversely
  `TROCHOID_DS_M` is **not** bookkeeping: the C1 assertion (60 deg) and the kappa-based time
  re-parameterisation both read the RESAMPLED polyline, so it owns a hard validity cliff at
  `ds_t ~ 0.0098 m` on `fixture_A` (60.3 deg at 0.010; frozen 0.004 sits **2.45x inside** at
  45.4 of 60 deg) above which `fixture_A` dies to a generator `AssertionError` (20/20 harness
  errors, 0 episodes) while `fixture_B` — 3.4x looser, narrower patch, fewer row-end turns —
  degrades PHYSICALLY (19/20, covc 0.9500, `stick_frac` 61x, `slip_m` 3.8x, `escaped_frac` 0.05).
  The predicted `ds_base` ceiling 0.0314 is CONFIRMED (no C1 trip there; the sagitta never bites).
- Verdict: **DISCARD of the axis as a lever** — no arm beats the champion, `keep=false` on every
  cell (B saturates 20/20 on both arms, Fisher p = 1.0), and p < 0.01 is arithmetically
  unreachable on an axis whose candidate equals its own baseline. Payload kept as rig facts.
- **Not claimed:** no replacement law for `max_turn_deg(ds_t)` (non-monotone: 73.0 deg at 0.012
  then 66.4 deg at 0.014 — resample vertex-phase aliasing against the row-end semicircles; only
  the bracket is claimed). No novelty, no learned component, no `novelty_score`, no synthetic
  proxy.
- **Axis status: CLOSED (both halves).** Section 0's claim that "the measurement chain has no
  unvaried frozen term left that anyone has named" was true only because nobody ran the header
  audit; N532 named the last one and closed it. N198 and N478d remain director-owned
  (`open_needs_director`, NOT queued). A next iteration must name a different axis.
- Evidence: `results/aegis_v2/N532_*.jsonl` (9 files), equations.md row N532, graph
  `N532_path_discretisation` + `N532 -> N533_next_axis`.

### N534 — the STRAY RAIL `WS_LIMIT_M = 0.60` — PRE-REGISTERED (before any dose was run)
- Why (audit, not taste): after N532 closed the path-discretisation axis and N533 closed the servo
  damping ratio, `WS_LIMIT_M` is the last never-dosed literal on the physics path. `CELL_M` is
  BARRED (`FINE_M = CELL_M/2` IS the metric pitch); `CONTACT_C` is certified inert by N533's
  6-decade pre-probe; `KP_PRESS` 1.0 multiplies an already-saturated press; `LAUNCH_Z_M` 0.02 is
  a launch COUNTER threshold (measurement only); `RESIDUAL_CLIP_M` has the residual off in 798/815
  archived arms.
- WHAT IT IS (read the source, do not guess): (a) TERMINATION — `norm(cur[:2] - axis) >
  WS_LIMIT_M` flags the episode `escaped`, control STOPS being applied and the remaining scrub
  ticks are counted WITHOUT contacts; (b) the saturation value of the tracking-error accumulator
  `err`, which feeds `slip_m = p90(err)` — but `SWEEP_TOL_M = 0.035 << 0.60`, so that branch is
  reachable only for already-escaped ticks. It is a SAFETY ENVELOPE / termination threshold, not a
  controller gain, and it is **plan-independent** (no waypoint, chase or press term reads it), so
  coverage can move ONLY through termination.
- Delta: `AEGIS_WS_LIMIT_M` registered in `KNOB_GLOBALS` (env read, default the frozen 0.60) +
  header field `ws_limit_m` / `sweep_tol_m` + one new PURE OBSERVATION `max_exc_m` (largest
  non-escaped excursion; no control path reads it). Candidate arm = dose, paired baseline arm =
  `--compare-env AEGIS_WS_LIMIT_M=0.60`, same 20 seeds (G4). Cells `0,0`, `0.03,6`, `0.32,64`,
  `0.40,80`; doses `0.30 / 0.45 / 0.90 / 1.50`.
- **CONFOUND PRE-REGISTERED (states the only honest reading before any number exists):** a coverage
  GAIN at a larger radius cannot be a capability claim — it means the run kept simulating instead
  of terminating. Only a LOSS at a smaller radius is a physical cost. `escaped_frac` and
  `max_exc_m` are reported in every arm so the confound is visible, not hidden.
- Falsifiers, pre-registered:
  - **N534.1 NULL/INERTNESS** — at `0,0`, where `escaped_frac = 0`, `coverage_cont` must be EXACTLY
    invariant over the whole dose range (delta 0.0000, p > 0.9). If it moves anywhere with zero
    escapes, WS_LIMIT_M is doing something other than terminating and THE AXIS IS VOID.
  - **N534.2 CONTAINMENT** — a rail is a safety envelope only if it lies OUTSIDE the reachable set
    of a legitimate head. Predicted reachable set at `0,0`: the scored patch corner is
    `hypot(0.20, 0.12) = 0.233 m` and `r_eff <= 0.045 m`, so `p100(max_exc) <= 0.28 m` and the
    frozen 0.60 holds >= 2.1x margin, making 0.30 a near-binding and 0.45 a non-binding dose.
  - **N534.3 SLIP IS NOT CROSS-DOSE COMPARABLE** — escaped ticks append `WS_LIMIT_M` into `err`,
    so `slip_m` inherits the dose itself in any arm with escapes. Only `escaped_frac = 0` arms have
    comparable slip; a slip "improvement" at a bigger radius is the dose echoing back.
  - **N534.4 TERMINATION COST** — any dose BELOW the reachable excursion must cost coverage. If
    0.30 is inert at `0,0` per N534.2, it is predicted inert; the first dose predicted to BIND is
    ~0.22-0.28, not 0.30.
  - **N534.5 NO FRESH AIR AT THE FRONTIER** — at `0.32,64` / `0.40,80` a larger radius must NOT
    rescue coverage: those escapes are divergences (the head is commanded off the patch), so
    continuing to actuate drives it further away, not back. Predicted `coverage_cont(1.50) <=
    coverage_cont(0.60)` on B. A material RISE would be a termination artefact and is barred from
    being a keep by the confound above.
- Kill rule: if `|delta coverage_cont| <= 0.001` in EVERY arm on every cell -> DISCARD the lever,
  KEEP the containment certificate (the rail is non-binding over a 5x dose range and the reachable
  set is measured, not asserted). No `novelty_score`, no learned component, no synthetic proxy,
  segment 15, gate untouched (post-hoc tag), metric untouched.

### N537 — the FORCE CEILING `F_CLAMP_N` as a SINGLE factor — PRE-REGISTERED (before any dose)
- Why (audit, not taste): `AEGIS_F_CLAMP_N` is registered in `KNOB_GLOBALS` and its archived
  dose history is **exactly three files, all from N210's coupled 3x3 grid at `SIM_HZ = 1920`**
  (`AEGIS_SIM_HZ=1920,AEGIS_PRESS_M=0.4,...,AEGIS_CONTACT_K=20000,AEGIS_F_CLAMP_N=15.0` and the
  40.0 arm). It has **never been dosed as a single factor against the champion at the frozen
  240 Hz**, and N210 moved `PRESS_M` and `CONTACT_K` with it, so the rail was never isolated from
  the press it caps. N533.4 then concluded from measurement that "the N214 escape floor is a
  force-rail floor, not a missing-damping floor" — i.e. the audit that closed the servo axis
  pointed at this literal and no iteration dosed it.
- WHAT IT IS (read the source): a HARD CEILING on the total commanded force magnitude, applied
  AFTER the wall term, the press term and the PD chase have all been summed:
  `if mag > F_CLAMP_N: F *= F_CLAMP_N/mag; f_clamp_ticks += 1`. It is plan-independent (no
  waypoint or path term reads it), so it can move coverage only through the achievable contact
  force. `f_clamp_ticks` and `f_cmd_max_n` are ALREADY logged per episode, so the binding
  fraction is measured, not inferred. Frozen 3.0 N; the AEGIS spec band is 10-25 N, so the
  frozen rail caps the band by 3.3-8.3x (the N210/N199 arithmetic-gap argument).
- Delta: `AEGIS_F_CLAMP_N` only. No rig scored quantity, gate, metric, segment or seed is touched;
  the press law, path, coverage kernel and termination rail are byte-identical.
- Candidate arm = dose; paired baseline arm = `--compare-env AEGIS_F_CLAMP_N=3.0` on the SAME 20
  seeds (G4). Cells `0,0` and `0.32,64` (the frontier cell where the champion loses the head).
- **CONFOUND PRE-REGISTERED (the only honest reading before any number exists):** a coverage GAIN
  at a larger ceiling is NOT a capability claim by itself — the same 3 N that caps the normal
  force also caps the lateral chase authority, so raising the rail buys BOTH grip and the ability
  to yank the head harder, and the two are not separable by this knob. `f_clamp_ticks`,
  `f_cmd_max_n`, `fn_mean` and `escaped_frac` are reported in every arm so the confound is
  visible. A gain accompanied by a material RISE in `escaped_frac` is a rail artefact and is
  barred from being a keep by this clause.
- Falsifiers, pre-registered:
  - **N537.1 NULL AT CEILING** — at `0,0` the champion is already at `coverage_cont` 1.0000 /
    20-of-20, so the metric is SATURATED there and no dose can raise it. A dose that MOVES
    `0,0` DOWNWARD is a measured cost of the rail; a dose that leaves it exactly 1.0000 says the
    ceiling is not the binder on the certified stack.
  - **N537.2 BINDING FRACTION FALLS MONOTONICALLY** — `f_clamp_ticks` must be non-increasing in
    the dose at both cells and must reach 0 at a dose above the measured `p100(f_cmd_max_n)`.
    Refuted if the binding fraction is flat or rises over the dose range (the rail would then not
    be a ceiling on this stack).
  - **N537.3 THE RAIL IS NOT THE FRONTIER** — at `0.32,64` the champion loses the head by
    ESCAPE (N479: 15 of 20 escapes), and a head that has left the patch is not recoverable by
    more force — the PD chase would have to drag it back across the face. Predicted
    `coverage_cont(25.0) <= coverage_cont(3.0)` on B, i.e. **no gain at the frontier**. A
    material RISE would mean the escapes were force-starved, which would be a genuine mechanism
    finding; it would still have to survive the `escaped_frac` confound clause above.
  - **N537.4 NO FREE LUNCH DOWNWARD** — a dose BELOW the frozen 3.0 N must cost coverage at
    `0.32,64` if the ceiling binds at all. Predicted a monotone loss at 1.0 N. If 1.0 N is inert
    at the frontier too, the axis is doubly void (neither up nor down moves the metric).
- Kill rule: if `|delta coverage_cont| <= 0.001` at `0.32,64` in EVERY dose -> DISCARD the lever,
  KEEP the measured binding-fraction certificate (how much of the frozen champion's commanded
  force is actually clipped). No `novelty_score`, no learned component, no synthetic proxy,
  segment 15, gate untouched (post-hoc tag), metric untouched.

### N537 — RESULT (run 561): DISCARD (lever) / certificate KEPT + N533.4 scope-corrected
- `F_CLAMP_N` isolated as a single factor for the first time; 8 paired invocations, 960 physical
  episodes, 20 seeds x {A,B,R} x 2 arms x {`0,0`; `0.32,64`}, 0 harness errors.
- **CERTIFICATE (kept, quote from now on):** the frozen 3.0 N rail binds **1.0 tick per 400 on the
  certified B stack (0.25% of the budget)** and 23.55 (5.9%) at `0.32,64`. Unclipping it to 25 N
  removes 100% of the binding and moves `coverage_cont` by **+0.0024** (Welch p 0.8948, Fisher p 1.0,
  B `0/20` in every arm), with escapes 15 -> 13. **The AEGIS 10-25 N force band is NOT the binding
  constraint on `transfer_success`** — the unclipped command reaches 75.4 N and coverage does not
  respond. The constraint is the pose-noise ESCAPE channel (N479), which is a divergence, not a
  force deficit. **N533.4 is scope-corrected, not retracted:** the rail explains the KD = 0 escape
  ONSET (`f_cmd_max` 10.56 N vs the 3.0 N ceiling, 396/400 ticks clamped) but not the frontier ceiling.
- Also recorded: `f_clamp_ticks` is monotone on B but **NON-monotone on fixture_A** (33.65 at 6.0 N vs
  25.30 at the frozen 3.0 N) — the ceiling bounds the TOTAL command, so a dose inside the band can be
  clipped more often than one below it. Do not smooth this away.
- Verdict **DISCARD** of the axis as a lever: `keep=false` on all 8 cells; the metric is saturated at
  `0,0` and B reads 0/20 at every dose at the only dynamic-range cell, so the G4 keep is unreachable
  in BOTH directions. Evidence `results/aegis_v2/N537_clamp*_cell*.jsonl` (8 files), equations.md §N537.
- **Axis status: CLOSED.** With N532 (mesh), N533 (damping), N534 (abort radius), N535 (tick),
  N536 (wall) and N537 (force ceiling), every frozen literal on the physics path that any archived
  run touched is now audited as a single factor. The only never-dosed literals left are
  `AEGIS_RESIDUAL_CLIP_M`, `AEGIS_GATE_THETA` and `AEGIS_GATE_PROP_MIN`, and all three own nothing on
  the primary metric BY CONSTRUCTION (the residual is off by default; the gate is a post-hoc tag since
  run 170 and its rethink family closed in run 241). **A next iteration must name a different axis or
  ask the director to open one.** N198 and N478d remain director-owned (`open_needs_director`, NOT
  queued).

### N562 — WHAT IS THE ELONGATED-FACE REGISTRATION RESIDUAL? (attribution) + DOES IT COST ANYTHING? (frontier probe) — PRE-REGISTERED (before any dose of this iteration)
- Why this axis (and it is a measurement axis, not a lever): run 453 named the **ESTIMATOR**, not the
  path, as the binding floor (N453b). Run 496 (N478c) then measured the open number and could not name
  it: on `fixture_B` the registered centre still misses the true face centre by `reg_err_xy` p50
  5.6-8.1 mm / p90 13.6-16.3 mm, **invariant to the ray pitch `n` (32..128) and to the window slack
  `H` (0.6/0.9)**, and yaw-coherent (`corr(reg_err, reg_yaw_plan)` +0.14 -> +0.99). N478d (still
  director-owned, still NOT queued) named two candidate CELLS. Run 561 wrote a **read-only** diagnostic
  block into the rig source (`N561` header, lines ~2377-2432) that measures both cells plus a third
  nobody named, and pre-registered D1-D5 there -- but run 561 then spent its budget on N537 and never
  logged a result for it. This iteration EXECUTES that pre-registered measurement with its own paired
  runs and answers it, and adds the one thing every audit so far has failed to ask: **does the residual
  cost anything on the primary metric at all?** The certified frontier is `B = 1.00` at `1.60,320`
  (N478c) and **no archived run has ever probed a cell above it**.
- Delta: **NONE of the mechanism.** Every candidate arm is the certified N478 stack verbatim
  (`AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=32 AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0
  AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 --path trochoid`); the paired baseline arm is `--compare trochoid`
  with `--compare-env AEGIS_REG_CASTS=1.0,AEGIS_REG_CN=0.0,AEGIS_REG_DFACT=1.5,AEGIS_REG_N=32.0`
  (the frozen single-cast default the segment's whole wall was measured on), same 20 seeds (G4).
  The D1-D5 diagnostic fields are already in the source, are pure observations of quantities the
  estimator computes, and **no control path, scoring term, gate, metric, segment or seed reads any
  of them** (`rg_diag` writes into the episode record only).
- Phase A (attribution, 20 seeds x {A,B,R} x 2 arms per cell): `0,0`; `0.32,64`; `0.96,192` at
  `n = 32`; `0.96,192` at `n = 128` (D3); `0.96,192` with `AEGIS_POSE_FIX_U/V = 0` (D4).
- Phase B (cost probe, paired, 20 seeds x {A,B,R} x 2 arms): `1.60,320` / `2.00,400` / `2.40,480` /
  `3.20,640` -- the first measurement anywhere above the certified `1.60,320` wall.
- Falsifiers, pre-registered (D1-D5 verbatim from the source block, D6/D7 added here):
  - **D1** cell (i) ALIVE iff (`reg_band_z_span` > 1e-4 OR `reg_band_nbody` > 1 OR |`reg_bbox_du`| > 1
    pitch OR |`reg_bbox_dv`| > 1 pitch) on ANY seed. **If instead `reg_z_gap2 >= 0.010` and
    `reg_band_nbody == 1` on 20/20, cell (i) is FALSIFIED for this geometry.** NOTE recorded in
    advance: `reg_band_nbody` counts distinct **rigid bodies**, and the tank rim belongs to the SAME
    body as the top face, so `nbody == 1` does **not** by itself prove the band admitted one face --
    a **coplanar** out-of-face region would read `nbody = 1` AND `reg_band_z_span ~ 0`. The
    discriminating statistic is therefore `reg_bbox_du/_dv` (accepted oriented-bbox extents minus
    the frozen face extents `0.68 x 0.28 m`), and D1 is decided by it.
  - **D2** cell (ii) ALIVE iff `reg_branch_margin < 10 deg` OR `reg_branch_flips > 0` on any seed.
    Predicted FALSIFIED (N478c: folded plan-yaw residual 0.36 deg p90 against a 320 deg yaw sigma).
  - **D3** the estimator's own sampling term owns it IFF `reg_err_xy` p50 FALLS >= 2x from `n = 32`
    to `n = 128` (pitch 38.71 -> 9.45 mm). A flat p90 with a falling p50 CONFIRMS D3 (the residual
    was the TAIL); a flat p50 REFUTES D3 and leaves the residual UNNAMED.
  - **D4** placement, not geometry: with the plan-frame offset held at zero and the yaw dose kept,
    `fixture_B` p50 must fall below 2 mm. No fall => the residual is a property of the ray/face
    geometry alone.
  - **D5 NO LEVER IS CLAIMED from the attribution.** Every Phase A arm is the certified stack or its
    paired single-cast baseline; the attribution must not move coverage. A coverage GAIN would be read
    as a lever only with a paired p < 0.01 certificate.
  - **D6 COST (new, this iteration).** If `fixture_B` stays `20/20` at `3.20,640` -- i.e. at a pose
    noise **5x above** the certified wall with a 192 deg yaw sigma doubled again -- then the 5-16 mm
    estimator residual **costs nothing on the primary metric** and the estimator axis closes as a
    certificate. The first Phase B cell that drops below `20/20` is the **cost frontier**, and its
    `reg_err_xy` / `reg_border_frac` / `escaped_frac` counters say WHICH channel binds there.
  - **D7 PREDICTION, stated before any number exists.** The escape channel (N479, N537: a divergence,
    not a force deficit) binds **before** the estimator residual does, because the residual is 5-16 mm
    against a `0.035 m` contact radius (0.14-0.46 of `r_eff`) while a `0.32 m` plan-frame error
    already drives `coverage_cont` to ~0.02. Predicted Phase B order: `1.60,320` 20/20,
    `2.00,400` and above NOT 20/20, with `reg_err_xy` p50 still in the 5-16 mm band on the failing
    cells. If `reg_err_xy` instead GROWS by an order of magnitude at the first failing cell, the
    estimator owns the frontier and the mask lever (restrict the accepted cloud to the top-face
    support) becomes the next node.
- Kill rule: no coverage gain is claimed from any Phase A arm (D5). Status vocabulary is
  keep | discard | crash | unvalidated | invalid; a mechanism-free attribution can only be a
  **certificate** (`discard` of any lever, payload kept), so a `keep` is reachable ONLY from Phase B
  and ONLY if a cell is certified at `>= 0.90` where the paired baseline is not, with rig `keep=true`.
- Cost: 9 paired invocations, ~1800 episodes, < 400 s. Evidence dir `results/aegis_v2/N562_*`.
- No `novelty_score`, no learned component, no synthetic proxy, segment 15, gate untouched
  (post-hoc tag), metric untouched, no arithmetic coverage, no teleport.

### N562 — ANSWERED, run 562 — **KEEP** (frontier `1.60,320` -> `3.20,640`, 10.0x) + attribution certificate
- 9 paired invocations, 2160 physical episodes, 0 harness errors, rig `keep=true` on 6 cells.
- **KEEP**: `fixture_B` **20/20 at `3.20 m / 640 deg`**, `coverage_cont` 0.9906, Welch p 1.13e-33,
  Fisher p 1.45e-11 vs the paired single-cast default (0/20). `coverage_cont` is FLAT in `sigma_t`
  (0.9797 / 0.9844 / 0.9867 / 0.9813 / 0.9906 at `0.96,192 / 1.60,320 / 2.00,400 / 2.40,480 /
  3.20,640`), candidate `reg_ok` never drops below 20/20 while the baseline collapses
  20 -> 9 -> 3 -> 2 -> 1 -> 0 of 20. **10.0x** the original `0.32,64` wall, 2.0x the N478c wall.
- **D1 FALSIFIED / D2 FALSIFIED / D3 REFUTED / D4 REFUTED / D5 HELD / D6 CONFIRMED / D7 REFUTED.**
  All three N478d cells are dead. `experiments/N562_probe_band.py` (rig's own world, noise 0, H=0.6,
  n=32) accepts 123/123 hits, 0 rejected, 1 rigid body, and in the frame-CORRECT projection the
  oriented bbox is 0.675 x 0.278 m against the frozen 0.68 x 0.28 m face — 0.13 / 0.04 of one ray
  pitch. The 1 cm band admits exactly the flat top face.
- **THE RESIDUAL IS NAMED AND IT IS THE FROZEN `mean` CENTROID.** In the probe, at zero pose noise
  with the grid centred on the face, the **oriented-bbox midpoint error is exactly 0.0 mm** while the
  **centroid error is 7.71 mm**, and 118 of 123 accepted points are mirror-symmetric about the face
  centre. The displacement is carried by the 5 asymmetric corner sites. It is present at ZERO
  planning noise, flat in `sigma_t` to 3.20 m, only 1.45x responsive to a 4x ray-pitch cut, and
  **costs nothing on the primary metric** (D6: B 20/20 at 10x the wall with p50 still 6.16 mm).
- **DO NOT QUOTE the `*_du` / `*_dv` axis-resolved fields** of the N561 diagnostic. They use
  `_q = _P @ _R.T` with `_R = [[c,-s],[s,c]]`, which yields `P.(c,-s)` and `P.(s,c)` — the MIRRORED
  frame, not `(major, minor)`. Mirrored axes read 0.668 x 0.444 m against a 0.68 x 0.28 m face; the
  correct projection `P @ R` reads 0.675 x 0.278 m. Inline DO-NOT-QUOTE warning added at the dict;
  `N562_probe_band.py` is the ground truth. The frame-invariant fields (`reg_pitch_m`,
  `reg_band_z_span`, `reg_band_nbody`, `reg_border_frac`, `reg_pca_ratio`,
  `reg_branch_margin_deg`, `reg_branch_flips`, `reg_est_gap_mm`) are unaffected. Nothing scored
  reads the dict. `_pct`'s empty-sample guard was also fixed (at `sigma_t >= 2.40` every `fixture_B`
  episode returns `reg_err_xy = NaN` and the summary died after the episodes were written);
  default-OFF regression: 60 paired episodes, **0 field diffs**.
- Process note: the first D4 arm set `AEGIS_POSE_FIX_U/V=0`, which IS the frozen default (additive on
  the sampled draw), so it was a silent no-op. Logged as `N562_A5_MISSPCEEDED_noop.jsonl` and
  replaced by `AEGIS_POSE_NOISE=0,192` (XY noise exactly 0, yaw sigma 192 kept) — which is what D4
  actually meant. **A hold-at-zero offset needs `AEGIS_POSE_NOISE=0,<yaw>`, not `POSE_FIX=0`.**

### N563 — `AEGIS_REG_EST=extent` on the TRUNCATION-FREE lattice stack — **DONE** (run 564/565/566 DISCARD, cert kept)
- Why this is new and not a replay of N190/N191: both of those dosed the extent midpoint in the
  **TRUNCATING** `H = 0.35` window, where the frozen mean carries an `O(sigma)` truncation bias
  (N190: extent alone was INERT, `reg_err` 43.35 -> 36.57 mm, B delta -0.0070, p 0.923; N191: the
  2-point support midpoint LOST, B covc 0.9125 vs 0.9153, p 0.928). The cast lattice (N192) removed
  truncation entirely, and N562 measured the truncation-free regime directly: `reg_border_frac` =
  **0.0000** at every frontier cell and the extent midpoint is **exact** (0.0 mm) while the mean is
  **7.71 mm** off. **That cell has never been dosed.** It is an estimator choice on the
  already-certified sensor, not a new mechanism, not the retired G3 energy/SE(3)/flow family.
- Delta: `AEGIS_REG_EST` only (already in `KNOB_GLOBALS`, default 0 = the frozen mean). Stack is the
  certified N478 stack verbatim (`AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=32
  AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 --path trochoid`); the
  paired baseline arm is `--compare trochoid --compare-env AEGIS_REG_EST=0.0` on the SAME 20 seeds
  (G4), so only the estimator differs.
- **Pre-register BEFORE running.** (P1) `reg_err_xy` p50 on `fixture_B` must fall by >= 2x from the
  frozen mean (7.6-8.1 mm -> <= 4 mm) — a smaller fall means the extent midpoint is carrying its own
  one-pitch extremal noise (N191's failure mode) and the lever is dead. (P2) `fixture_B`
  `transfer_success` must not regress at `3.20,640` and `0.96,192`. (P3) If (P1) holds and B does not
  regress, the estimator residual is **closed as removed**, and the only remaining question becomes
  whether a `0 mm` residual buys a further frontier cell — probe `4.00,800` and `4.80,960` on the
  `extent` arm alone, and report `reg_ok` (the baseline arm already fails at `0/20` by `3.20,640`, so
  the lattice containment — not the centroid — is what the frontier is made of). (P4) Any gain is a
  keep ONLY with rig `keep=true` (>= 20 seeds, B > 0.70, paired p < 0.01, no coverage regression);
  an estimator arm that ties the frozen mean on B at every cell is a **certificate, not a keep**.
- Cost: 4 paired invocations, ~480 episodes, < 200 s. Evidence dir `results/aegis_v2/N563_*`.
- No `novelty_score`, no learned component, no synthetic proxy, segment 15, gate untouched
  (post-hoc tag), metric untouched, no arithmetic coverage, no teleport. N198 / N478d stay
  director-owned (`open_needs_director`, NOT queued).

### N679 — the pose-noise frontier TERMINATES ON SENSOR COST, NOT ON CAPABILITY — PRE-REGISTERED (written before any cell of this iteration ran)
- **Why this is new and NOT another infill.** Runs 604-645 were 42 "frontier infill" keeps: each one
  certified the certified stack at the next higher `sigma_t` and reported `B transfer_success` 1.00 at
  every cell, marching the pose-noise axis out to `16.00 m / 3200 deg` on a fixture whose tank major
  extent is 0.68 m (23x the object). Every one of them measured the SAME thing — that the cast lattice
  re-finds the face — and NONE of them asked where the axis STOPS. The chain has no wall in it, so the
  last published number is an extrapolation with no bound. This iteration does not add a cell; it asks
  the axis's only unasked question: **is there a capability wall, and if not, what terminates it?**
  Measured facts that make the question sharp: `AEGIS_REG_CASTS=0` derives
  `k = ceil(6 sigma_t/d)+1` from the LABEL, `d = DFACT*a_slack = 2.0*0.2319 = 0.46377 m`
  (`a_slack = H - rho_inf = 0.6 - 0.368`, cross-checked against the archived `reg_casts`: 43264 = 208^2
  at `16.00` and 20736 = 144^2 at `11.00`, both exact), so the cast budget is `k^2 ~ sigma_t^2`.
  `16.00,3200` already costs 43264 cast offsets and 559 s for 240 episodes on this rig.
- **Delta: NONE.** Certified N478/N562 stack verbatim
  (`AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=32 AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0
  AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 --path trochoid`), paired baseline arm
  `--compare trochoid --compare-env AEGIS_REG_CASTS=1.0,AEGIS_REG_CN=0.0,AEGIS_REG_DFACT=1.5,AEGIS_REG_N=32.0`
  on the SAME 20 seeds (G4). This iteration only READS `reg_casts`, `reg_ok`, `coverage_cont`,
  `wall-clock` and the champion contrast. No control path, scoring term, gate, metric, segment or seed moves.
- **PRE-REGISTERED FALSIFIERS (before any cell runs).**
  - **T1 CONTAINMENT HOLDS (capability wall does NOT exist in range).** `fixture_B` `reg_ok` 20/20 and
    `transfer_success` 1.00 at every ladder cell up to the last affordable one. If this fails at any
    cell, that cell IS the capability wall and it is reported as the terminator, not a keep.
  - **T2 THE COST LAW IS EXACT.** `reg_casts == ceil(6*sigma_t/0.46377)+1` squared, to the ray, at every
    cell: 67600 (k=260) at `20.00`, 132496 (k=364) at `28.00`, 173056 (k=416) at `32.00`. A cell whose
    measured `reg_casts` disagrees is reported with the MEASURED number, never the law's prediction.
  - **T3 THE AXIS IS COST-LIMITED.** Wall-clock per episode is proportional to `reg_casts` (report the
    measured sec/episode at each cell and the ray budget per cell). The axis terminates where the ray
    budget makes the 20-seed paired decider unaffordable, and the break-even `sigma_t` is reported for a
    NAMED ray budget rather than an implicit one.
  - **T4 NO DOWNWARD CLAUSE.** If a cell is 20/20 but paired mean `coverage_cont` REGRESSES vs the
    archived `16.00,3200` cell (0.989) by more than 0.005, that cell cannot claim the frontier moved.
  - **T5 KEEP BAR (G4).** A cell is a KEEP only with its OWN file's `compare.keep == true`
    (>= 20 seeds, `fixture_B transfer_success` > 0.70, paired p < 0.01, no coverage regression).
- **Cost:** one 3-seed screen to locate the terminator, then ONE 20-seed paired decider at the located
  cell. `timeout 1200` per invocation, anchored reap after each. Evidence `results/aegis_v2/N679_*`.
- **What a KEEP does and does not mean.** A KEEP certifies that the LOCALIZATION MECHANISM's dynamic
  range has no capability wall inside the measured range and that the axis terminates on SENSOR COST.
  Per N564.3 it is NOT a calibration-accuracy spec and must never be quoted as one: the paper reports
  `fixture_B` success with `reg_casts` and seconds/episode attached, or not at all.
- No `novelty_score`, no learned component, no synthetic proxy, segment 15, gate untouched (post-hoc
  tag), metric untouched (`fixture_B_transfer_success`), no arithmetic coverage, no teleport.
  N198 / N478d / N478e stay director-owned (`open_needs_director`, NOT queued).

### N679 — **ANSWERED, runs 679/680 — KEEP: NO CAPABILITY WALL; THE AXIS TERMINATES ON SENSOR COST at `sigma_t = 23.01 m`**
- **T1 HELD / T2 CONFIRMED TO THE RAY / T3 CONFIRMED / T4 HELD / T5 MET.** 1 screening row (r679,
  3 seeds, `32.00,6400`) + ONE 20-seed paired decider (r680, `20.00,4000`, 240 eps, 454 s,
  0 harness errors). Delta NONE throughout.
- **Decider:** `fixture_B transfer_success` **20/20 = 1.000**, `coverage_cont` **0.9867** (std 0.0265) vs
  the paired single-cast default **0/20, coverage_cont 0.0000**; Welch p **1.53e-31**, Fisher p
  **1.45e-11**, rig `compare.keep == true`. `reg_ok` 20/20 vs 0/20. A 2/20 (round face collapses at every
  archived cell too), R 11/20. Realized |plan offset| mean **21.63 m**, max **49.77 m** = **31.9x** the
  0.68 m tank major extent. Evidence `results/aegis_v2/N679_D_2000.jsonl`.
- **The cost law, zero free parameters** (`equations.md` ROW N679): rig-logged `reg_casts` =
  `(ceil(6 sigma_t/0.463768)+1)^2` at 20736 / 43264 / 67600 / 172225 for `sigma_t` = 11.00 / 16.00 /
  20.00 / 32.00 — **exact at all four cells**, with `d = 2.0 * a_slack` and
  `a_slack = H - rho_inf = 0.6 - 0.368132 = 0.231868 m`. `k ∝ sigma_t` to 0.24 %, so `R ∝ sigma_t^2`.
- **The wall, in closed form:** wall-clock is affine in the cast budget at **112.8 µs per cast offset**
  (107.35–117.98, ±4.7 %, over an 8.30x range of `k^2`), so the 1200 s / 120-candidate-episode budget
  affords `k^2_max = 88652`, `k_max = 297.7`, **`sigma_t_max = 23.01 m = 33.8x` the tank**.
  `22.00` is affordable (81796, 1107 s); `23.00` is not (89401, 1210 s).
- **What it retires and what it opens.** It closes the 42-cell infill chain (runs 604-645) with a stated
  bound instead of an unbounded extrapolation to `16.00 m`. It also names the ONLY lever that buys
  `sigma_t` without paying `k^2`: the lattice pitch `d` (i.e. `a_slack`, i.e. `REG_HALF_M`/`rho_inf`) and
  the two-stage selection `REG_CN` — cost scales as `(6 sigma_t/d)^2`, so a 2x coarser `d` buys 4x the
  frontier for the same seconds. That is the next node, and it is NOT queued without the director.
- N198 / N478d / N478e stay director-owned (`open_needs_director`, NOT queued).

### N565 — the CONTAINMENT CLIFF of the lattice frontier: is the certified frontier a PROPERTY OF THE DRAW? — PRE-REGISTERED (before any dose of this iteration)
- **Why this axis (audit of the load-bearing law, not a lever).** The whole frontier claim rests on
  ONE analytic sentence written into the rig at N195.1: with `d = DFACT*a_slack = 2a`, the lattice
  `k = ceil(6 sigma_t/d)+1` gives a cast-union square of half-extent
  `W = (k-1)d/2 >= 3 sigma_t`, and "SOME lattice offset lands inside `C(f) = f + [-a,a]^2` for every
  planning error with `|e|_inf <= W`". Measured from the archive (read-only, this iteration):
  `W/sigma_t` = 3.001 / 3.013 / 3.020 / 3.005 at `sigma_t` = 4.80 / 6.40 / 8.00 / 16.00 — the law is
  exactly right. **But `pose_noise` is an UNBOUNDED Gaussian (N213), so `|e|_inf <= W = 3 sigma_t` is a
  statement about the DRAW, not about `sigma_t`, and NO archived frontier row has ever reported its own
  realised `|e|_inf/sigma_t`.** Every certified cell in runs 604-645 uses ONE seed base
  (`AEGIS_BASE_SEED=111000`), whose 20 `fixture_B` draws have `max|e|_inf/sigma_t = 2.118` — a 30%
  margin — and the realised offsets are `sigma_t`-scaled copies of each other, so the SAME 20 numbers
  certify every cell from `0.32` to `16.00 m`. **The frontier certificate is therefore a per-seed-base
  Bernoulli, not a capability, and its failure probability is computable and has never been stated:**
  `P(fail) = 1 - (1 - 2(1-Phi(W/sigma_t)))^20 = 5.1%` at `W/sigma_t = 3.02`.
- **What is new here.** Nothing in the archive varies the DRAW at fixed mechanism: `AEGIS_BASE_SEED` has
  never been a dose (every run 1-645 reads one seed base), so the containment law has never been tested
  at its own boundary. This iteration moves `max|e|_inf/sigma_t` across the boundary `W/sigma_t ~ 3.02`
  at FIXED mechanism and FIXED cell, which is the only experiment that can refute sufficiency,
  necessity, or both.
- **Delta: NONE of the mechanism.** Every arm is the certified N478/N562 lattice stack verbatim
  (`AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=32 AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0
  AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 --path trochoid`); the paired baseline arm is `--compare trochoid`
  with `--compare-env AEGIS_REG_CASTS=1.0,AEGIS_REG_CN=0.0,AEGIS_REG_DFACT=1.5,AEGIS_REG_N=32.0`
  (the frozen single-cast default the segment's wall was measured on), SAME seeds inside each run (G4).
  The only dosed variable is `AEGIS_BASE_SEED`, which is the DRAW (`pose_noise = gauss` per seed, plus
  friction / tool / customer draw) and touches no knob, no plan term, no force, no score, no gate.
  Seed bases are chosen OFFLINE by replaying the rig's own documented draw
  (`random.Random(seed*104729+3).gauss(0, sigma)`; `seed = base + 1000*(suite_index*seeds) + k`) and
  every claim is then re-verified against the `pose_noise` the RIG LOGS per episode, never against the
  offline replay.
- **Arms** (cell `4.00,800`, 20 seeds x {A,B,R}, `k = 53`, `k^2 = 2809` casts, `W = 12.08 m`,
  `W/sigma_t = 3.020`): bases `111000` (archive, `z_inf = 2.118`), `117280` (3.017, inside by 0.1%),
  `141600` (3.523, 1 seed out), `147800` (3.542, 2 seeds out), `100100` (3.549, 2 seeds out). Cell
  `16.00,3200` (`W/sigma_t = 3.005`) at bases `111000` and `141600` for the scale-invariance test.
- **PRE-REGISTERED FALSIFIERS (stated before any cell runs).**
  - **P1 SUFFICIENCY.** Every base with `z_inf <= W/sigma_t` must hold `fixture_B` 20/20. REFUTED if
    any in-window base drops an episode -> the frontier is not containment-limited and the law is wrong.
  - **P2 NECESSITY (the cliff).** Every base with `z_inf > W/sigma_t` must FAIL exactly the seeds with
    `z_inf > W/sigma_t` — predicted `B` success `20 - n_over` = 19/20, 18/20, 18/20 — and those episodes
    must show `reg_err_xy` far outside the certified 5.6-8.1 mm p50 band or `escaped_frac ~ 1`.
    REFUTED if an out-of-window base still reads 20/20 -> the frontier is bought by something else
    (partial-face acceptance / `reg_cn` margin) and N195.1's sentence is not the operative law.
  - **P3 ATTRIBUTION.** Failing episodes must be identified by the rig's OWN logged `pose_noise`
    (`max|dx|,|dy| > W`), not by the offline replay; a mismatch voids the run.
  - **P4 SCALE INVARIANCE.** The criterion is on `z = e/sigma_t`, so the FAILING SEED SET must be the
    SAME at `16.00,3200` as at `4.00,800` for the same base (only the `ceil` jitter of `W/sigma_t`,
    3.020 vs 3.005, can move a borderline seed). REFUTED if the sets differ -> the frontier is a
    physical rather than a statistical limit, which is a strictly stronger claim than N562's D6.
  - **P5 THE QUANTUM.** Report `P(fail)` per cell from the realised `W/sigma_t` and state the
    certificate rule: a frontier cell may be quoted ONLY with its realised `max|e|_inf/sigma_t` and the
    margin `1 - z_inf/(W/sigma_t)`. This is the payload of the iteration regardless of P1-P4.
- **Kill rule / status.** A mechanism-free law test can only be a certificate (`discard` of any lever,
  payload kept). A `keep` is reachable ONLY if a candidate arm certifies a `fixture_B` cell at
  `>= 0.90` where the paired baseline is not, with the rig's own `compare.keep == true`.
- Cost: 7 paired invocations, ~840 physical episodes, < 900 s. Evidence dir `results/aegis_v2/N565_*`.
- No `novelty_score`, no learned component, no synthetic proxy, segment 15, gate untouched (post-hoc
  tag), metric untouched, no arithmetic coverage, no teleport, no coverage estimated from anything but
  physics contact points. N198 / N478d stay director-owned (`open_needs_director`, NOT queued).

### N565 — ANSWERED, runs 752/753 — **P4 CONFIRMED (scale invariance) + P5 quantum stated; N565 closed except the unlogged P1-111000@4.00,800 archive control**
- **P4 (run 752, base 141600, cell `16.00,3200`, fresh 278 s paired 20x3x2, 0 harness errors, rig `keep=true`).**
  Candidate B `19/20` (`coverage_cont` 0.9383) vs paired single-cast default `0/20` (Welch p 9.07e-14,
  Fisher p 3.05e-10). Failing set `{161618}` IDENTICAL to the `4.00,800` cell (run 721); rig-logged
  `z_inf = 3.5234` at both cells; out-of-window (`56.37 m > W = 48.00 m`, `W/sigma_t = 3.0000` exact);
  failing episode `reg_ok=False`, escaped, `covc 0.0000`. The frontier is a property of the DRAW:
  the same seed index fails at 4x the label because the realized offsets are sigma-scaled copies.
  Fresh run reproduces the archived (previously unlogged) `N565_b141600_1600.jsonl` exactly.
- **P4 control (run 753, base 111000 at `16.00,3200`, fresh 275 s paired, rig `keep=true`).** Candidate B
  `20/20` (`covc` 0.9836) vs `0/20` (Welch p 2.45e-31, Fisher p 1.45e-11). Realized `max z_inf = 2.1179`
  (29.4% margin); `reg_err_xy` p50 6.52 / p90 14.93 mm inside the certified band. Reproduces
  `N565_b111000_1600.jsonl`. **P5 quantum:** per-20-seed-cell `P(fail) = 10.3%` at `W/sigma_t = 3.00`;
  frontier cells quotable only with realized `max|e|_inf/sigma_t` + margin. Evidence
  `results/aegis_v2/N565_P4_1600_b{141600,111000}.jsonl`, equations.md ROW N565-P4.
- Director iter-22's FACC-E refused in the same iteration as the 63rd G3 replay (run 751, INVALID).
  N198 / N478d stay director-owned (`open_needs_director`, NOT queued).

### N565 — CLOSED (run 797): the last open pre-registration, P1 archive control, is ANSWERED
- **Run 797, base 111000 at `4.00,800`, fresh paired 20x3x2 (120 episodes, 0 harness errors, rig `keep=true`).**
  Candidate B `20/20` (`coverage_cont` 0.9898) vs paired single-cast default `0/20` (Welch p `1.906e-33`,
  Fisher p `1.451e-11`); A `0.20`, R `0.55`. `k = 53`, `reg_casts 2809` exact, `W = 12.079833 m`,
  `W/sigma_t = 3.019958`, realized `max z_inf = 2.117885` (margin `+29.87 %`), `reg_ok 20/20`, 0 escapes,
  `reg_err_xy` p50 `8.11` / p90 `16.55 mm`. Reproduces the previously unlogged archive control
  `results/aegis_v2/N565_b111000_400.jsonl` bit-identically.
- **N565 P1/P2/P3/P4/P5 all answered -> N565 CLOSED.** Evidence `results/aegis_v2/N565_P1_400_b111000.jsonl`,
  equations.md ROW N565-P1-ARCHIVE, JSONL run 797, graph node `N565-P1-archive-797`.
- Director iter-68's FACC (SE(3) deformable affordance + EBM energy attention) refused in the same
  iteration as the 106th G3 replay. **Queue remains EMPTY**; N198 / N478d / N478e stay
  `open_needs_director` (NOT queued) — the director must open one of them or close segment 15.

