# N481 — footprint-aware row plan x tool-head size (run 499, segment 15)

Written BEFORE the first rig invocation of this idea (G7).

## Why this cell (audit, not proposal)

- The director's iter-23 proposal (FACC-SE(3): force-adaptive contact control, SE(3)-conditioned,
  energy-based in-context contact attention, flow-matching -> affordance-equivalence bridge) is the
  **58th replay of the G3 permanently retired family** (eq78 / N70-N76 / manifold-switch /
  flow-bridge / energy-gated / EBM / SE3 / DAEF / FACC / SEAFC / VMEAF / SEAF). G4 unpairable:
  `PATH_MODES` carries 0 injectable SE3/energy/affordance/FACC hook. G7: predicted-only 78 pts is
  not evidence. Refused, 0 episodes, 0 rig bytes.
- Stall break by MEASUREMENT (run-453/476/494-498 precedent). Audit of all 984 archived rig
  headers x 52 `KNOB_GLOBALS` keys found two knobs at literally **0 doses ever**:
  `AEGIS_BASE_DS_M` and `AEGIS_RESIDUAL_CLIP_M` (the latter inert while `RESIDUAL_ACTIVE=0`).
  Neither owns the primary metric, so instead of dosing an inert knob this iteration crosses two
  MEASURED axes that have never been crossed:
  - **path geometry** x **tool-head footprint**. `AEGIS_PAD_HU_M/HV_M` were laddered by N214
    (run 363) on `trochoid` and `raster` ONLY; `fitted` (I10/I21, run 238 KEEP) and `fitro`
    (I18, run 239) were run at DEFAULT pad ONLY. Archive cross-tab: 0 files with
    `path_mode in {fitted, fitro}` AND a non-default `pad_hu_m`.

## The measured gap

N214 (run 363), `trochoid`, square pad, pose noise `0,0`, 20 seeds x 3 suites:

| pad r_eff | fixture_B transfer_success | coverage_cont | escapes |
|---|---|---|---|
| 0.035 (frozen) | 20/20 | 1.0000 | 0 |
| 0.025 | 20/20 | 0.9062 | 0 |
| 0.0175 | **0/20** | 0.6414 | 1 |
| 0.010 | **0/20** | 0.3836 | 3 |

and at `0.03,6`: `trochoid` + pad 0.0175 = **0/20** (coverage 0.5227), frozen pad = 15/20.

So the N214 head floor `r_eff >= 0.0275 m` was measured **only on the champion's row plan**
(rows at `CELL_M = 0.05` pitch -> covering radius 0.025 m). `fitted` recomputes its row count from
the SAME pad (`spec_in["r_eff"] = min(half)` is passed in at line 1398), so it has never been
asked the question the floor is written in.

## Pre-registered hypothesis (written first)

Covering-radius law (the one N214 already bracketed):

    a cell is covered iff some contact lies within r_eff of its centre, so the path's local
    covering radius must satisfy  rho_path <= r_eff - eps,   eps = tracking/scoring slack.

- `trochoid`: rho = CELL_M/2 = 0.025 m. N214 measured the floor at r_eff* in (0.025, 0.0275],
  i.e. eps in [0.0000, 0.0025] plus the run-to-run bracket -> eps_meas = 0.0025..0.0075.
- `fitted`:  n_rows = ceil(side / (2 r_eff)), pitch = side / n_rows <= 2 r_eff,
  rho = pitch/2 <= r_eff. On fixture_B (side 0.12 m) at r_eff 0.0175 -> n_rows 4, pitch 0.03,
  rho = 0.015 m; at 0.010 -> n_rows 6, pitch 0.02, rho = 0.010 m (rho == r_eff, zero margin).

| id | outcome | statement |
|---|---|---|
| **H1** confirmatory | at least one pad in {0.0175, 0.020, 0.0225}: `fitted` fixture_B transfer_success >= 0.70 AND > paired `trochoid` on the same 20 seeds, Fisher p < 0.01, mean-coverage delta >= -0.005 -> rig `keep=true` -> **KEEP** |
| **H2** refuting | `fitted` fixture_B <= 0.70 at EVERY pad >= 0.010 while the paired `trochoid` is what N214 measured -> the N214 floor is **not** a path-footprint mismatch: the binding channel is dynamic (slip/stall), so **DISCARD** and the floor stays a tool property |
| **H3** control | `fitted` at the DEFAULT pad must not regress: cite the archived paired record `I21_r238_fitted_vs_trochoid_n00.jsonl` (fitted B 20/20 cov 1.0000 vs trochoid 20/20 cov 0.9453). No new run; if a new run were needed it would be at pad 0.035 only |
| **H4** dose | the NEW floor reported as a bracket, never a point: the smallest tested pad where `fitted` still reads B >= 0.90 together with the largest where it reads < 0.70. Champion floor 0.0275 is the reference; a >= 1.5x downward move (floor <= 0.018) is the headline |
| **H5** composition | at pose noise `0.03,6` with pad 0.0175, `fitted` beats the paired `trochoid` (archived 0/20) with Fisher p < 0.01 -> the floor move survives a Tier-2 planning error |
| **H6** arm-isolation control | `fitro` (fitted rows + trochoid loops) is run at pad 0.0175 as the loop-excitation variant; if `fitro` < `fitted` on B, the loops cost coverage at small footprint and `fitted` alone is the shipped candidate |

**KILL RULE**: H2 at all pads -> status `discard`, no keep claimed, N214's floor is left where
N214 put it.

## What is NOT claimed

- No novelty of mechanism: `fitted`/`fitro` are the rig's own shipped path modes (I10/I18/I21);
  the contribution is the never-run cross and the refutation-or-confirmation of N214's floor
  attribution.
- No rig byte changes (`rig_md5` must equal `ce401a0293b695b7b56c066024c1003f`), no teleport,
  no coverage/success arithmetic (every `coverage_cont` and `success` comes from physics contacts
  at return time), no gate / segment / metric change, nothing read from
  `benchmarks/restroom_sim.py`, nothing from a predicted score.
- No learned component; edge budget untouched (no GPU, no downloads).

## Cells (all 20 seeds x 3 suites x 2 paired arms, `timeout 1200`, anchored cleanup)

| # | env | --path | pose noise | file |
|---|---|---|---|---|
| 1 | `AEGIS_PAD_HU_M=0.0175 AEGIS_PAD_HV_M=0.0175` | fitted vs trochoid | 0,0 | `results/aegis_v2/N481_r499_fitted_pad0175_n00.jsonl` |
| 2 | pad 0.020 sq | fitted vs trochoid | 0,0 | `.../N481_r499_fitted_pad0200_n00.jsonl` |
| 3 | pad 0.0225 sq | fitted vs trochoid | 0,0 | `.../N481_r499_fitted_pad0225_n00.jsonl` |
| 4 | pad 0.010 sq | fitted vs trochoid | 0,0 | `.../N481_r499_fitted_pad0100_n00.jsonl` |
| 5 | pad 0.0175 sq | fitro vs trochoid | 0,0 | `.../N481_r499_fitro_pad0175_n00.jsonl` |
| 6 | pad 0.0175 sq | fitted vs trochoid | 0.03,6 | `.../N481_r499_fitted_pad0175_n0306.jsonl` |

Pad comes from the process env, so both arms see the SAME footprint (only `--path` differs);
`--compare trochoid` pairs on identical seeds -> identical friction/tool/noise/customer (G4).
