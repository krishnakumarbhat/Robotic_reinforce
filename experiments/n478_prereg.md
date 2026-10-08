# N478 — the k x k CAST LATTICE is the next estimator lever, and the derived-k GUARD is off by a factor of 3

Pre-registered 2026-10-02, BEFORE any N478 rig invocation. Source: ideas.md section 4, new row
queued at run 477 (the queue was empty; N477 was the only `queued` row and run 477 answered it).
Rig: experiments/kaggle_aegis_sweep.py v2, PyBullet DIRECT, system python3, CPU.
**Delta: NONE -- 0 rig bytes changed.** Every arm is env + `--compare-env` only.

Director proposal for this slot (FACC iter-2, "Deformable-Affordance FACC with Energy-Based Contact
Gate", predicted 76) is REFUSED before execution: it is the **44th** replay of the
G3-permanently-retired eq78 / N70-N76 / manifold-switch / flow-bridge / energy-gated / EBM /
SE(3)-affordance / DAEF / FACC family; G4 unpairable (PATH_MODES carries 0 injectable
SE3/energy/manifold/FACC-head hook, RESIDUAL_HOOK_FN only); G7 predicted-only 76 is invalid for a keep.
Its stated FALSIFIER ("abort if <5% gap on force-perturbed split") is also unpairable in this rig.

## Where the frontier is after run 477 (N477)
N477 moved the certified boundary from `0.32,64` to **`0.64 m / 128 deg`** (B = 0.733 at 60 seeds,
Welch p 3.35e-08, Fisher p 1.74e-09, rig keep=true) and established the mechanism: the controlled
variable is the **ray pitch p = 2H/(n-1)**, not the extent H. Champion knob set is now
`H=1.2, n=65` (p = 0.0375 m). The wall is `0.72,144`.

Residual at the certified edge: median `reg_err_xy` = 11.4 mm (B) and the N477 note that the face
rays are a *lower bound* (the `r_eff` disc, N211) understates what the sensor actually samples.
The registration window is still a **single** cast: `AEGIS_REG_CASTS=1` is frozen, so the N192/N195
k x k lattice -- the one mechanism built to kill the truncation crescent -- has NEVER been switched on
in the entire run history.

## The defect this run tests (closed-form, from the rig source)
`_cast` samples the top face through ONE window of half-extent `H` centred on the NOISY pose. The face
itself has worst-yaw half-extent `rho_inf = hypot(0.34, 0.14) = 0.3677 m`. So the single window
CONTAINS the whole face iff the 2-axis planning error satisfies `|e|_inf <= a_slack := H - rho_inf`.

- At the champion `H = 1.2`: `a_slack = 0.8323` m, lattice pitch `d = 1.5 * a_slack = 1.2484` m.
- Pose noise is 2-axis Gaussian, `e ~ N(0, sigma_t) I_2`, so the shipped AEGIS convention is
  `E = 3 sigma_t`. Containment of a single window is then a ~35% event at `0.64,128` and ~23% at
  `0.72,144` (computed, not estimated) -- i.e. **the majority of episodes at the new certified edge
  score a TRUNCATED face**, and a truncated face biases both the mean centroid and the PCA yaw.

**The guard is the defect.** The derived-k law in the rig reads

    k = ceil(6*sig_t / d) + 1   if (REG_CASTS == 0 and sig_t > a_slack) else max(int(REG_CASTS), 1)

but the N195.1 containment condition it claims to implement is `d/2 <= a_slack` AND
`(k-1)d/2 >= E` with **`E = 3*sig_t`, not `sig_t`**. The guard tests `sig_t > a_slack` where the
derivation needs `3*sig_t > a_slack`. At the champion `H=1.2` we have `a_slack = 0.8323`, so for every
cell in the band (`sigma_t` in {0.40 ... 0.72}) the guard is FALSE and the shipped law returns
**k = 1** -- the lattice is switched off precisely where it is needed. Minimum legal k by the
N195.1 condition is 3 / 4 / 5 / 5 at `0.40,80` / `0.48,96` / `0.64,128` / `0.72,144`.

This is the same SHAPE of finding as I22 (`ROW_CENTRE`: the band sat `CELL_M/2` low because a closed
form put rows on cell edges) -- a shipped constant that is right by a factor, not a new mechanism.

## Pre-registered hypotheses (fixed before the first rig invocation)
- **P1 (truncation, primary).** At `0.64,128` with `H=1.2, n=65` frozen, raising
  `AEGIS_REG_CASTS` from 1 to a legal k lowers the median `reg_err_xy` on fixture_B. Mechanism claim
  is falsified if median `reg_err_xy` does not move.
- **P2 (boundary move).** If some k gives `B > 0.70` at `0.72,144` with paired Welch
  `p(coverage_cont) < 0.01` AND Fisher `p(success) < 0.01` vs the paired k=1 arm, and the rig's own
  `keep` field is true, the certified boundary moves `0.64,128 -> 0.72,144`.
- **P3 (guard defect, the sharpest and cheapest).** `AEGIS_REG_CASTS=0` (the shipped derived-k law)
  is **bit-identical to k=1** at every cell in the band, because the guard `sig_t > a_slack` is false
  for all of them. Falsifier: if the `REG_CASTS=0` arm differs from k=1 anywhere, my reading of the
  guard is wrong and the mechanism story is retracted.
- **P4 (dose shape).** `B` is non-decreasing in k over {1, 3, 5, 7} at `0.64,128`, with a plateau at
  k >= 5 (k_min = 5 by N195.1). Falsifier: a peak at k=3 with k=5 worse means the coarse-stage
  argmax, not truncation, is the binder.
- **P5 (falsifier readout, cheap).** The WINNER's own border margin (`reg_cn_margin`, already logged
  by the rig) must satisfy `reg_cn_margin >= a_slack`; a containing window has that margin by
  construction. If an arm raises B while its median `reg_cn_margin < a_slack`, it kept a CUT window
  and the truncation claim is refuted for that arm.
- **P6 (kill rule).** If NO k arm beats k=1 with paired p < 0.01 at `0.64,128`, the registration
  estimator family is CLOSED and the residual at `0.64,128` is the controller / contact, not the
  estimator. The boundary stays `0.64,128` and the next frontier moves off the estimator entirely.

## Design (G4 paired on every arm; same 20 seeds -> same friction / tool / noise / customer draw)
Both arms: `--path trochoid --compare trochoid --seeds 20 --suites fixture_A,fixture_B,fixture_R`,
`AEGIS_REG=depth` + `AEGIS_ROW_CENTRE=1` carried in the process env (so the compare arm inherits
them), `AEGIS_REG_HALF_M=1.2`, `AEGIS_REG_N=65`, `AEGIS_POSE_NOISE=<cell>`.
Candidate arm varies ONLY `AEGIS_REG_CASTS` (with `AEGIS_REG_CN=16` coarse-to-fine, whose per-cast
cost is 256 rays so k=7 is 17k rays, not 49*65^2 = 207k).
Paired compare arm = the frozen champion, `AEGIS_REG_CASTS=1`.
Cells: `0.64,128` (certified edge, primary) and `0.72,144` (the N477 wall, P2).
Readouts: per-suite `success`, `coverage_cont` + the rig `compare` record (Welch / paired / Fisher),
`reg_err_xy` median, `reg_ok`, `reg_casts`, `reg_cn_margin`, `escaped`.

## Decision rule (fixed now)
- **KEEP** only if P1 and P5 hold AND some k gives B > 0.70 at `0.72,144` with paired Welch
  `p(coverage_cont) < 0.01`, Fisher `p(success) < 0.01`, >= 20 seeds, and rig `keep: true`.
  Then the certified boundary moves to `0.72,144` and the guard defect (P3) is reported.
- **DISCARD (mechanism answer)** if P6 fires: the estimator is closed, boundary stays `0.64,128`.
- **crash** if the rig errors twice (G5).

## Cost / budget
<= 12 rig invocations, ~10k rays per registration episode, CPU only, no GPU, no download.
Every invocation wrapped in `timeout 1200` (G5) and reaped with the anchored pattern.
