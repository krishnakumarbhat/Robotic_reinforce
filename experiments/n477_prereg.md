# N477 — window dose ABOVE 0.6: is 0.32 m / 64 deg the wall of the WINDOW or of its CONTACT-STARVATION?

Pre-registered 2026-10-02, BEFORE any N477 rig invocation. Source: ideas.md §4 row N477 (queued at run 476).
Rig: experiments/kaggle_aegis_sweep.py v2, PyBullet DIRECT, system python3, CPU. Delta: NONE (no rig bytes changed).
Director proposal for this slot (FACC iter-25, energy-based in-context attention replacing the flow-mat head,
predicted 74/100) is REFUSED before execution: it is the 43rd replay of the G3-permanently-retired
eq78 / N70-N76 / manifold-switch / flow-bridge / energy-gated / EBM / SE3-affordance / DAEF / FACC family;
G4 unpairable (PATH_MODES carries 0 injectable SE3/energy/manifold/FACC-head hook, RESIDUAL_HOOK_FN only);
G7 predicted-only 74pts is invalid for a keep. N477 is the only `queued` row and it executes instead.

## Mechanism under test
The sensor casts ONE +-H ray grid (`AEGIS_REG_CASTS=1` frozen, so the k x k lattice of N192 is OFF). Two
H-dependent quantities move together when H is dosed:
  (E) EXTENT: half-width of the sampled region. Containment of the +-inf face (rho_inf = 0.3677 m)
      requires a_slack = H - rho_inf > 0, i.e. H > 0.3677. a_slack(0.6)=0.232, a_slack(0.9)=0.532,
      a_slack(1.2)=0.832.
  (S) STARVATION: ray pitch = 2H/(n-1) at frozen n=32, i.e. 0.035 m at H=0.6 but 0.045 m at H=0.9 and
      0.075 m at H=1.2. The number of rays that land ON the face is (face area in window)/(2H/(n-1))^2,
      so raising H at fixed n SHRINKS the face sample by (0.6/H)^2: 1.00x -> 0.44x -> 0.25x.
H is therefore NOT a monotone dose a priori; N190.6 already recorded H=0.9 LOSING to H=0.6 at
0.20 m / 40 deg for exactly this reason. N477 asks whether that trade is what caps the certified boundary.

## Pre-registered hypotheses
P1 (monotone dose, ideas.md primary): B transfer_success at 0.40,80 is non-decreasing in H over
    {0.6, 0.9, 1.2}: 0.70 -> >= 0.70.
P2 (boundary move): if B > 0.70 at 0.40,80 for some H with paired Welch p(coverage_cont) < 0.01 and
    Fisher p(success) < 0.01 vs the paired H=0.6 arm, the certified boundary moves past 0.32 m / 64 deg.
P3 (kill rule, ideas.md): any arm below B = 0.70 at 0.32,64 => the WINDOW family is CLOSED and the
    residual is the controller, not the estimator.
P4 (starvation discriminator, NEW, this run): at FIXED H=1.2, raise n so the ray pitch matches the
    H=0.6 arm (n=65 -> 0.0375 m, 4.1x the rays). If B at 0.40,80 recovers toward the H=0.6 level, the
    wall is (S) contact starvation and the extent is free. If B does not move, the wall is (E) the
    extent/truncation crescent. Falsifier: n=110 at H=1.2 would be 12x rays; n=65 is the cheap
    pitch-matched arm and is the one run.

## Design
Every arm: 20 seeds x 3 suites (A, B, R), trochoid, steps 400, AEGIS_REG=depth + AEGIS_ROW_CENTRE=1 in
BOTH arms (G4 paired: same seeds -> same friction / tool / noise / customer draw).
  Cells: pose noise 0.40,80 and 0.48,96 (the two cells N476 placed at B 0.70 and 0.60), plus 0.32,64 for P3.
  Doses: H = 0.6 (reference), 0.9, 1.2. Candidate arm H, paired compare arm H=0.6 (0.6 cell paired to the
  frozen 0.35 champion knob set so the ladder re-measures its own base).
  Secondary: H=1.2, n=65 at 0.40,80 paired to H=0.6,n=32 (P4).
Readouts: per-suite success, coverage_cont (+ Welch/paired/Fisher from the rig `compare` record),
reg_err_xy median, escapes.

## Decision rule (fixed now)
- KEEP only if some H in {0.9,1.2} gives B > 0.70 at 0.40,80 with paired p < 0.01 AND P3 passes AND the
  rig's own `keep` field is true. Then Tier-2 extends to 0.40 m / 80 deg.
- DISCARD-family (report as discard with the mechanism answer) if P1 refutes and P4 attributes the cap to
  (E) extent: the estimator is CLOSED at H=0.6 as a PLATEAU PEAK, the boundary stays 0.32 m / 64 deg, and
  the frontier moves to the controller.
- CRASH if any run errors twice.