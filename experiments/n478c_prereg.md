# PRE-REG / N478c (2026-10-02, segment 15) — spend the N478b saving on `n`, the parity term

## Refusal first (binding, before any rig byte is touched)
Director iter-18 proposal **FACC — Force-Adaptive Contact Control, SE(3)-conditioned**
(energy-based contact field + adaptive compliance; flow-matching expert replaced; predicted 78;
"flow-matching -> affordance-equivalence" bridge) is **REFUSED: 55th replay** of the
G3-permanently-retired family (eq78 / N70-N76 / manifold-switch / flow-bridge / energy-gated /
EBM / SE3 / DAEF / FACC / SEAFC / VMEAF). Label rotation ("FACC", "FACC-SE3", "FACC-E",
"VMEAF", "adaptive compliance") does not change the mechanism class: a fixed-or-learned
affordance manifold scored by an energy and bridged to a flow-matching expert IS the retired
manifold/flow-bridge class. G4: `PATH_MODES` carries **0** injectable SE3/energy/manifold/FACC
hook (only `RESIDUAL_HOOK_FN`, I3-owned), so no paired 20-seed Fixture-B arm is even expressible.
G7: predicted-only 78 pts is not evidence. **0 episodes, 0 rig bytes** for the proposal.

## Stall break (run-453 / run-476 / run-494 / run-495 precedent)
Refuse the replay, then spend the iteration on the ONE `queued` row: **N478c**, proposed by
run 495.

## N478c question
N194/N195 named the SURVIVING error of the whole lattice as the ray pitch of the ONE kept
cast, `2H/(n-1) = 38.71 mm` at `n = 32` — not the selection (N478b closed that: `cn >= 12`,
`DFACT = 2.0`, 3.6x-6.5x fewer rays, frontier extended to `0.96 m / 192 deg` at `B = 1.00`).
Run 495 freed the rays without touching `n`. So: **is `n` the binding term?**

## Why this is a real question and not a formality
Once N192's lattice picks a window that CONTAINS the whole top face, the frozen MEAN estimator is
unbiased (N195.1: a containing window has border margin `>= a_slack`), so the truncation bias that
drove N190's p90 (14.6 mm at 1 cm/2 deg -> 118.8 mm at 16 cm/32 deg) should be gone. If it is gone,
the residual should be set by the sampling of the face on the ray grid, i.e. by `2H/(n-1)` — the
term N194 named. If instead the residual is already converged at `n = 32`, then the parity term is
NOT the binder and the frontier question moves to pose noise itself. Both readings are real; the
experiment distinguishes them and the difference is worth a run either way.

## Frozen stack (identical to every N478/N478b cell)
`AEGIS_PATH=trochoid AEGIS_ROW_CENTRE=1 AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=<dose>
AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0`
Paired baseline `--compare trochoid --compare-env AEGIS_REG_CASTS=0.0,AEGIS_REG_CN=0.0,
AEGIS_REG_DFACT=1.5,AEGIS_REG_N=32.0` = the certified N478 arm (`AEGIS_REG_N=32` is written into
the compare env explicitly, because the compare arm inherits the process env otherwise — the
compare record's `baseline_knobs` is the audit).
20 seeds x 3 suites x 2 paired arms per cell. Only `AEGIS_REG_N` is dosed (already in
`KNOB_GLOBALS`). 0 rig bytes changed.

Derived quantities, all arithmetic in the code path (`rays = k^2 cn^2 + n^2`, `k = ceil(6 sigma/d)+1`,
`d = DFACT * a_slack`, `a_slack = 0.6 - hypot(0.34,0.14) = 0.23234 m`):

| sigma_t | k | parity 2H/(n-1) at n=32 / 64 / 128 | rays base (n=32, DFACT 1.5) | rays candidate n=32/64/128 (cn=12, DFACT 2.0) |
|---|---|---|---|---|
| 0.32 | 6 | 38.7 / 19.0 / 9.4 mm | 36 864 | 6 208 / 7 312 / 11 552 |
| 0.96 | 14 | 38.7 / 19.0 / 9.4 mm | 331 776 | 29 248 / 32 288 / 44 608 |
| 1.12 | 16 | " | 458 752 | 37 888 / 40 928 / 53 248 |
| 1.28 | 18 | " | 607 104 | 47 680 / 50 720 / 63 040 |
| 1.60 | 22 | " | 887 296 | 70 720 / 73 760 / 86 080 |

Every candidate dose is at or below the `n = 32` cost the N478 arm already paid at the SAME cell,
so the run-495 saving funds the ladder outright.

## Pre-registered hypotheses (fixed BEFORE running)

- **Hc1 (parity-bound, confirmatory).** On fixture_B at `0.96,192`, `reg_err_xy` p90 falls
  monotonically along `n = 32 -> 48 -> 64 -> 96 -> 128` and `p90(n=128) <= 0.6 * p90(n=32)`;
  the drop is consistent with the centroid of a face sampled on a pitch `2H/(n-1)`, i.e.
  `err ~ pitch / sqrt(pts on face)`. Then `n` IS the binding term and the best `n` is carried into
  phase B.
- **Hc2 (refuting).** `reg_err_xy` p90 plateaus by `n = 48`/`64` (deltas within seed noise) at a
  floor well below the parity term. Then the parity term is NOT the binder, the estimator has
  already converged at `n = 32`, `n > 32` is INERT, and phase A is reported as a refutation of the
  N194/N195 "surviving error = ray pitch" attribution. Frontier then belongs to phase B alone.
- **Hc3 (no-regression guard, both readings).** At `0.96,192` fixture_B must hold
  `transfer_success = 1.00` and `|delta coverage_cont| <= 0.005` vs the paired certified arm at
  EVERY `n`; and at the older certified cell `0.32,64` the winning arm must not drop below `1.00`.
  A dose that regresses the metric is not a result.
- **Hc4 (frontier probe, run unconditionally).** With the cheapest certified arm, push
  `sigma_t` to `1.12 / 1.28 / 1.60 m` (yaw paired 224 / 256 / 320 deg), paired on the same 20 seeds.
  If fixture_B stays `1.00` (>= 0.90) for BOTH arms at `1.60,320`, the certified frontier is
  `>= 5.0x` the old `0.32 m / 64 deg` wall and N478b's `0.96,192` "frontier" is not the end of
  the axis. This is a CERTIFICATION question, not a p-value question: both arms saturated means
  `Fisher p = 1.0` by construction and that number will be reported as such, never as a win.
- **Hc5 (falsifier readout, logged not claimed).** `reg_cn_margin` (the winner's own border margin,
  metres) and `reg_casts` are logged every episode. Any episode whose winning margin `< 0` kept a
  CUT window, and `reg_err_xy` must show it. Run 495's `cn = 8` failure mode (3/40 episodes,
  `reg_err_xy ~ a_slack`, `coverage_cont = 0.0000`) must NOT reappear at `cn = 12` for any `n`.

## Kill rules
- Hc3 violated at any dose -> that dose is discarded, no frontier claim from it.
- `harness_errors > 0` or any episode `backend != pybullet` -> the cell is void, not a result.
- Two rig errors -> log `crash`, commit, exit (G5).
- Nothing here may be reported from `benchmarks/restroom_sim.py` (synthetic proxy) or from a
  predicted score (G2/G7).

## Statuses
`keep` requires a MEASURED fall (Hc1) or a MEASURED frontier extension (Hc4) on the physical
fixture_B transfer_success, >= 20 seeds, paired, with the compare record archived.
---

## PRE-REG 2 (written AFTER phase A/B, BEFORE these cells) — is the B floor TRUNCATION? dose `a_slack` post-lattice
Phase A returned: fixture_B `reg_err_xy` p90 is FLAT in `n` (13.63 / 16.08 / 14.96 / 15.46 / 16.26 mm
at n = 32/48/64/96/128 on ONE selection rule, cn=12 DFACT=2.0, k=14 identical every rung) while
fixture_A falls 7.9x (4.35 -> 0.55 mm) on the SAME ladder. So the ray pitch binds on A (round,
few points on face) and binds on NOTHING on B (elongated). On B the residual becomes MORE
yaw-coherent as the sampling noise dies (corr(err, reg_yaw_plan) = +0.14 / +0.77 / +0.94 / +0.92 /
+0.99 at n = 32/48/64/96/128): what is left is systematic and orientation-linked, i.e. a
candidate TRUNCATION bias of the winning window (the centroid of a window that clips one side of
an elongated, rotated face), not sampling noise. `a_slack = H - rho_inf` is the term that sets how
much face a window can clip, and it was last swept in N477 in the SINGLE-CAST regime (run 476-481)
— it has NEVER been swept with the lattice on.

- **Hd1 (truncation, confirmatory).** With the lattice on and ONLY `AEGIS_REG_HALF_M` dosed
  (0.6 -> 0.9 -> 1.2, so `a_slack` 0.2323 -> 0.5323 -> 0.8323 m, `d = 2 a_slack`, `k = ceil(6 sigma/d)+1`
  = 14 -> 7 -> 5 at 0.96,192), fixture_B `reg_err_xy` p50 FALLS monotonically and lands below the
  `n = 32` plateau (8.12 mm) at `H = 1.2`. Then truncation was the binder, `n` is inert, and `a_slack`
  is the next frontier lever.
- **Hd2 (refuting).** p50 stays inside the 5.6-8.1 mm band at all three H. Then the B floor is
  neither ray pitch (phase A) nor window truncation, and the residual is left unattributed — which
  is reported as such, with no replacement law claimed.
- **Hd3 (guard).** fixture_B `transfer_success = 1.00` and `|delta coverage_cont| <= 0.005` at
  every H vs the paired `H = 0.6` lattice arm on the same 20 seeds; `reg_ok = 20/20`.
- **Hd4 (inertness control, pre-registered as NOT a result).** Rays/episode COLLAPSE with H
  (coarser `d` -> smaller `k`): 29248 (H=0.6) -> ~14k (H=0.9) -> 4624 (H=1.2) at 0.96,192. If
  Hd1 holds with rays falling, the frontier gets cheaper AND better — but the cost side alone is
  not a result and will not be reported as one.
- Paired baseline: `--compare-env AEGIS_REG_CASTS=0.0,AEGIS_REG_CN=12.0,AEGIS_REG_DFACT=2.0,
  AEGIS_REG_N=32.0,AEGIS_REG_HALF_M=0.6` (the run-495 cheap certified arm, bit-identical control).
