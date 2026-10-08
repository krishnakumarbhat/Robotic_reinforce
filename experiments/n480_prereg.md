# PRE-REG / N480 (2026-10-02, segment 15) — DECOMPOSE the compound pose-noise label

## 1. Refusal first (binding, before any rig byte is touched)

Director iter-20 style proposals in the SE(3)/energy/flow/affordance-EBM class
(SE3-EAA / DAEF / EBM / SEAFC / VMEAF / FACC / eq78 / N70-N76 re-labelled with a new acronym) are
**REFUSED: 57th replay** of the G3-permanently-retired family. `PATH_MODES` carries **0** injectable
SE3/energy/affordance hook (the only hook is `RESIDUAL_HOOK_FN`, I3-owned); a predicted novelty
score is not evidence (G7). **0 episodes, 0 rig bytes** for the proposal.

## 2. Stall break (run-453 / run-476 / run-494 / run-495 / run-496 / run-497 precedent)

The queue is empty: N478, N478b, N478c, N478e are `done`, N478d and N198 are `open_needs_director`
and the director has not authorised either. So: audit the archived evidence for a factor that has
**never been separated**, and dose it. The audit (972 archived rig headers +
every `compare_env` in `results/aegis_v2/*.jsonl`) found exactly one such factor.

## 3. N480 question — the pose-noise axis is a COMPOUND LABEL and has only ever been dosed along a ray

`AEGIS_POSE_NOISE = "sigma_t_m,sigma_yaw_deg"` declares **two** factors. The archive shows the
segment has dosed the pair along the single ray `sigma_yaw_deg = 200 * sigma_t_m` and essentially
nowhere else:

| archived `pose_noise_cfg` | files | on the ray `deg = 200*m`? |
|---|---|---|
| `0,0` | 530 | (zero) |
| `0.03,6` / `0.01,2` / `0.02,4` / `0.005,1` / `0.05,10` / `0.08,16` / `0.12,24` | 93+92+22+17+6+3+3 | yes |
| `0.16,32` … `1.60,320` (the whole certified frontier) | 15+15+14+13+11+10+9+7+7+4+3+3+2+2+1+1+1 | yes |
| `3.2,640`, `2.00,400`, `2.56,512`, `6.4,1280`, `1.6,320` | 7+2+1+1+1 | yes |
| `0,8` `0,14` `0,18` `0,28` `0,56` `0,112` (N201, run 311, **yaw-only**, sigma_t = 0) | 13+2+2+2+2+2 | OFF the ray — the ONLY separated cells in 497 runs, and they stop at 112 deg |
| `2.0,2` `0.5,2` `0.1,2` `0.05,10` (translation-heavy one-offs) | 3+1+1+6 | off-ray, 1 cell each |

So the headline certified by runs 494-496 — **"fixture_B = 1.00 through 1.60 m / 320 deg"** — is a
**compound** number whose two halves have never been separated at the frontier. Three things follow
and all three are testable with the rig unchanged:

1. The lattice size is `k = ceil(6*sigma_t/d) + 1` — driven by **sigma_t only**
   (`experiments/kaggle_aegis_sweep.py:2110`). A compound label therefore pays the full `k^2` ray
   bill for a yaw error that the registration PCA branch then partly removes: on the elongated face
   `yaw_est` is observable and the folded plan residual is reported under 1.6 deg even at
   `sigma_yaw = 320` (N478c). **If the yaw half is free on B, the frontier is a translation number
   and the ray budget is being sized by a factor that does not own the metric.**
2. On the **round** face `yaw_est = None` and the FULL prior yaw error stays in the plan
   (same file, the `tank_shape == "elongated"` guard at line 2211). So the two halves must have
   **opposite signatures**: translation recoverable everywhere, yaw recoverable only on the
   elongated face. One number cannot describe both. The paper's own Tier-2 requirement
   ("< 0.005 m / 1 deg", I5) is a compound label for the same reason.
3. `AEGIS_POSE_NOISE` is env-only (not in `KNOB_GLOBALS`), so a decomposition needs one process per
   cell — cheap, and it keeps the rig at **0 bytes changed**.

## 4. Frozen stack (identical to every N478b/N478c cell; rig md5 must stay `ce401a02…`)

Candidate (the N478b-confirmed cheap lattice):
`AEGIS_PATH=trochoid AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=32 AEGIS_REG_CASTS=0
AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 AEGIS_ROW_CENTRE=1`, everything else default.
Paired baseline (G4): `--compare trochoid --compare-env
AEGIS_REG_CASTS=1.0,AEGIS_REG_CN=0.0,AEGIS_REG_DFACT=1.5,AEGIS_REG_N=32.0` = the **frozen
single-cast champion** (`REG_CASTS=1` -> `k=1`, the exact module default, the arm the segment's
whole pose-noise wall was measured on). Both arms read the SAME `AEGIS_POSE_NOISE` from the
process env, so the pairing is on identical noise draws, friction, tool and customer (G4).
20 seeds x 3 suites x 2 arms = 120 episodes per cell. `md5` of the rig checked before and after.

## 5. Pre-registered predictions (stated BEFORE any run)

- **D1 (attribution, the load-bearing one).** On `fixture_B` the compound cell `1.60,320` and the
  translation-only cell `1.60,0` must be **indistinguishable**: `|delta coverage_cont| <= 0.005`
  and `Fisher p >= 0.5` between them (both vs the single-cast baseline, which should collapse).
  If the compound is significantly WORSE than translation-only, the yaw half is NOT free and the
  "frontier is translation-bound" reading is REFUTED.
- **D2 (yaw is free on the elongated face).** Yaw-only `0,320` must hold `fixture_B` at
  `transfer_success >= 0.90`, `coverage_cont >= 0.98` (N201's `0,112` = 1.00 / 0.9867 extrapolated
  2.9x). REFUTED if `fixture_B` `transfer_success < 0.90` at `0,320`.
- **D3 (yaw is NOT free on the round face — the two-regime claim).** Yaw-only `0,64` must collapse
  `fixture_A` to `transfer_success <= 0.30` and `0,192` to `~0`, because `yaw_est = None` there.
  REFUTED if `fixture_A` holds >= 0.90 at `0,64`.
- **D4 (budget).** `reg_casts` must be 1 on every yaw-only cell (k is inert when `sigma_t = 0 <=
  a_slack`) and `ceil(6*sigma_t/d)+1` on the translation-only cells. Reported as ray counts; a
  yaw-free frontier is a cheaper frontier.
- **D5 (falsifier for D1/D2 together).** If translation-only `1.60,0` does NOT reach
  `fixture_B = 1.00`, the compound label is not decomposable in the direction claimed and the
  frontier stays a compound number.

## 6. Kill rule / bar

No gate, metric, segment or knob-default change. A KEEP requires `>= 20` seeds,
`fixture_B transfer_success > 0.70` and `p < 0.01` in the candidate's favour in the rig's own
`compare` record. The scientific deliverable is the **decomposition itself** (D1-D4), which stands
or falls on the numbers, not on the summary verdict. Nothing here is claimed for fixture_A or
fixture_R: they are reported as the cross-regime readout only (G2 — `fixture_B` is the metric).

## 7. Cost

Phase A 4 cells (translation-only `0.32,0` `0.96,0` `1.60,0` `2.56,0`) + Phase B 3 cells (yaw-only
`0,64` `0,192` `0,320`) + Phase C 2 compound controls (`0.32,64` `1.60,320`) = 9 invocations,
~1080 episodes, every one wrapped in `timeout 1200`, anchored-orphan reap after each.
Evidence dir `results/aegis_v2/N480_r498_*`.
