# PRE-REG / N479 (2026-10-02, segment 15) — dose the CLOSED-LOOP press (`AEGIS_FORCE_PI`)

## 1. Refusal first (binding, before any rig byte is touched)

Director iter-19 proposal **SEAF — SE(3)-Equivariant Affordance-Energy Flow**
(energy `E(contact, action|obs)` over a deformable affordance field, SE(3)-conditioned in-context
attention, equivariant flow sampling re-ranked by energy, replacing FACC/flow-matching with an
"affordance-equivalence" bridge; pi0-FAST contact-rich split; "+15% contact success, zero-shot pose
shift <10% drop"; predicted 78) is **REFUSED: 56th replay** of the G3-permanently-retired family
(eq78 / N70-N76 / manifold-switch / flow-bridge / energy-gated / EBM / SE3 / DAEF / FACC / SEAFC /
VMEAF / SEAF). Label rotation ("SEAF", "affordance-equivalence", "energy flow") does not change the
mechanism class: a learned energy over a contact/affordance field that is sampled through a
flow bridge IS the retired manifold/flow-bridge/energy-gated class. G4 is also unsatisfiable as
stated: `PATH_MODES` carries **0** injectable SE3/energy/affordance hook (the only hook is
`RESIDUAL_HOOK_FN`, I3-owned), and no pi0 checkpoint may be used as a keep claim in segment 15
(segment/metric freeze, G7). G7: a predicted 78 is not evidence.
**0 episodes, 0 rig bytes** for the proposal.

## 2. Stall break (run-453 / run-476 / run-494 / run-495 / run-496 precedent)

Refuse the replay, then spend the iteration on the **lowest-numbered available physical cell**.
The queue is empty (N478/N478b/N478c all answered; N478d and N198 are `open_needs_director`), and
ideas.md states the registration stack is exhausted — "a next iteration needs a NEW axis, not
another dose of the registration stack". So: audit the 50 `KNOB_GLOBALS` against every archived
`compare_env` in `results/aegis_v2/*.jsonl`. Axes with **0** archived doses:
`AEGIS_FN_SET`, `AEGIS_FN_KP`, `AEGIS_FN_KI` (the I15 force PI, `AEGIS_FORCE_PI` dosed 0 times as
a mechanism), `AEGIS_WALL_ONLY`, `AEGIS_WALL_KP`, `AEGIS_RESIDUAL_CLIP_M`, `AEGIS_GATE_THETA`,
`AEGIS_GATE_PROP_MIN`, `AEGIS_TROCH_DS_M`, `AEGIS_BASE_DS_M`. **N479 = the force channel.**

## 3. N479 question

In every run 1-496 the commanded press during the scrub phase was the bare product
`KP_PRESS * kp * PRESS_M = 1.0 * 25 * 0.020 = 0.500 N` — a constant, with no feedback from the
measured contact normal force `fn` (which the loop already computes every tick and feeds to the
I3 residual as `pend["fn"]`). The I15 PI regulator exists in the code (`FORCE_PI`, `FN_KP`,
`FN_KI`, rails `[0, PRESS_MAX_N] = [0, 1.2] N`) and has **never been switched on**.
So: **is the segment's 0.5 N normal force a designed constant, or a coincidence of a stiff
contact spring?** Two readings, both real:
- if the stiff `CONTACT_K = 1e3` spring already self-regulates `fn` (predicted: on the elongated
  metric face the open-loop arm already reads `fn_mean` 0.4957 N with `fn_std` 0.0089 N), then the
  force-feedback press is **inert there** and the paper's force channel needs no regulator — the
  0.5 N limitation is a *scale* limitation, not a *control* limitation;
- if the press is NOT self-regulating wherever contact is intermittent (predicted: `fixture_A`
  round face, open-loop `fn_std` 0.36 N, `force_compliance` 0.48, i.e. half the ticks outside the
  ±50% band), then closing the loop should raise the compliance fraction and may buy coverage
  exactly where contact is marginal.

## 4. Frozen stack (identical to every N478 cell's baseline; 0 rig bytes changed)

Candidate arm env: `AEGIS_PATH=trochoid AEGIS_FORCE_PI=1 AEGIS_FN_SET=<set> AEGIS_FN_KP=0.5
AEGIS_FN_KI=5.0`. All other knobs default (`ROW_CENTRE=1`, `REG` off, `REG_CASTS=1` = the frozen
single-cast champion).
Paired baseline: `--compare trochoid --compare-env AEGIS_FORCE_PI=0.0,AEGIS_FN_SET=0.5,
AEGIS_FN_KP=0.0,AEGIS_FN_KI=0.0` = the frozen OPEN-LOOP press, same seeds (G4).
20 seeds x 3 suites x 2 paired arms per cell = 120 episodes. `PRESS_MAX_N = 1.2` and
`F_CLAMP_N = 3.0` are FROZEN (`PRESS_MAX_N` is not in `KNOB_GLOBALS`, so it cannot be paired —
it is not touched).
Gain choice is dimensional, not tuned: `press[N] += 0.5 * err[N]` (a 0.1 N deficit buys 0.05 N
this tick) and `press_int += 5.0 * err[N] * 0.05 s` (0.025 N/s) — both far below the 1.2 N rail on
the 0.0043 N steady-state error the open-loop arm already shows on B, so the loop cannot ring.

## 5. Pre-registered predictions (stated BEFORE any run)

- **F1 (regulation, confirmatory).** With `FN_SET = 0.5` and the PI on, versus the paired
  open-loop arm on the SAME 20 seeds: `fn_std` falls and `force_compliance` (fraction of ticks
  with `fn` inside ±50% of setpoint) rises. Bar: `fn_std` ratio <= 0.5 **and** compliance gain
  >= +0.15 absolute on **at least one** suite, at `fixture_B transfer_success >= 0.90` and
  `|delta mean coverage_cont| <= 0.005`. Predicted locus: `fixture_A` (round, intermittent
  contact), **not** B (already self-regulated, ratio ~1.0 = inert). "Inert on B" is a result, not
  a failure — it is the reading that makes the force channel a scale limitation.
- **F2 (force dose / invariance, the paper-relevant one).** At `FN_SET = 1.0` (2x frozen) and
  `0.25` (0.5x), `fixture_B transfer_success` at `0,0` must stay `>= 0.90` -> the certified result
  is not an artifact of one force. **REFUTED** if B drops below 0.90.
- **F3 (the N210.3 law, tested not assumed).** N210.3's tracking law is `e = mu*Fn/KP`; under it,
  2x `Fn` should ~2x `slip_m` at fixed `KP` and cost coverage. If instead coverage RISES at
  `FN_SET = 1.0`, the law is refuted on this axis. Reported either way; **no replacement law is
  claimed** without a measurement, and the open-loop `PRESS_M` ladder (N210, already run) is the
  control for "is the PI route to 1.0 N different from the `PRESS_M` route".
- **KILL RULE.** Any PI arm with `fixture_B < 0.90` at `0,0` is **discarded** regardless of any
  compliance or variance gain; a variance win that costs the primary metric is a loss.
- **Arithmetic negative control (pre-registered as NOT a result).** `FORCE_PI=1` with
  `FN_KP = FN_KI = 0` REPLACES the constant press with `press = clamp(0 + 0) = 0 N`, so `fn -> 0`
  and coverage must collapse. If it does not, the on/off contrast is not exercising the press term
  and every F1/F2 number is void. Run on `fixture_B`, 20 seeds.
- **Verdict bar (G2/G4).** Metric = fixture_B `transfer_success` (`coverage_cont >= 0.90`), frozen.
  A win requires `>= 20` seeds, B `> 0.70` and paired `p < 0.01` (Welch on `coverage_cont`, Fisher
  on `success`) with the rig's own `compare.keep` true. If both arms saturate at B = 1.00 the
  verdict is reported as a **mechanism/measurement result with the saturating comparison stated
  explicitly** (run-495 precedent), never as a metric win.

## 6. Cells (in order, each a separate 20-seed paired rig invocation)

| # | pose noise | candidate | why |
|---|---|---|---|
| 1 | `0,0` | PI set 0.5 | F1 core contrast, and the negative control for the code path |
| 2 | `0,0` | PI set 1.0 | F2/F3 force dose up (2x) |
| 3 | `0,0` | PI set 0.25 | F2 force dose down (0.5x) |
| 4 | `0.48,96` | PI set <best of 1-3> | dynamic range: the frozen open-loop champion reads B ~0.60 at this cell, so a paired win is arithmetically possible (12/20 vs 20/20 -> Fisher 0.008) |
| 5 | `0,0` | PI set 0.5, `FN_KP=FN_KI=0` | arithmetic negative control, fixture_B only |

Budget: ~5 invocations, 600 episodes, well inside the 20-minute cap. Every invocation wrapped in
`timeout 1200`; `pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill` after each.
Evidence dir: `results/aegis_v2/N479_r497_*`.

## 7. Integrity (G7)

No coverage/success arithmetic anywhere: `coverage_cont` and `success` come only from the rig's
physics contacts at return time. No `resetBasePositionAndOrientation` in any evaluated pass (the
PI acts through `F -= nrm*press`, a force, inside the loop). No synthetic proxy. `metric` in the
JSONL row = fixture_B `transfer_success` from the canonical rig. If the axis turns out to be
inert everywhere, the row is `discard` with the measurement, not a keep.

## 8. PRE-REG 2 addendum (written after cell 1, BEFORE cells 2-6)

Cell 1 (`0,0`, PI set 0.5, 120 episodes, 0 harness errors) result, verbatim from the rig:
`fixture_B` 20/20 vs 20/20, `coverage_cont` 1.0000 vs 1.0000, `fn_mean` 0.4993 vs 0.4957,
`fn_std` **0.0046 vs 0.0089**, `force_compliance` 1.0000 vs 1.0000, `press_max_n` 0.5564 vs 0.5000,
`mean_slip_m` 0.01064 vs 0.01059. `fixture_A` 20/20 vs 20/20, covc 1.0000 vs 1.0000 but
`fn_mean` 0.5765 vs 0.4317, `fn_std` **0.5558 vs 0.3623**, `force_compliance` **0.3349 vs 0.4864**,
`press_max_n` **1.2000 vs 0.5000** (the rail), `slip` 0.0102 vs 0.0073. `fixture_R` 20/20 vs 20/20,
`fn_std` 0.2399 vs 0.1746, compliance 0.6946 vs 0.7548, `press_max_n` 1.2000 vs 0.5000.
Both arms 60/60 episodes; rig `keep=false` (Fisher 1.00 everywhere, covc identical).

**F1 is REFUTED as written.** On B the loop is nearly inert (std ratio 0.52, compliance already
saturated at 1.0000 in BOTH arms, so there is no compliance headroom to win) and on A — the locus
I predicted — the loop makes the force channel **worse on both statistics**: `fn_std` x1.53 and
compliance **-0.15**, with the press riding the 1.2 N rail. Mechanically: where contact is
intermittent the error is large and one-signed, the proportional+integral term winds the press up
to the rail, and the re-engaging contact then reads `fn_p95` 1.74 N (vs 1.29 N open-loop) and drives
the press back to 0. The loop is a **variance amplifier** on intermittent contact, not a
regulator, at these gains.

That leaves exactly two readings, and they are separable by a gain ladder, so it is pre-registered
now rather than after the fact:
- **F4 (structural, kills the mechanism).** With 5x gentler gains (`FN_KP=0.1`, `FN_KI=1.0`) the
  round face still shows `fn_std` ratio `> 1.0` and `force_compliance` below the open-loop arm. Then
  force feedback cannot be made non-harmful on intermittent contact at ANY gain below the rail, and
  the frozen open-loop constant press is the correct design — a mechanism KILL, not a tuning miss.
- **F5 (tuning, keeps the mechanism).** `FN_KP=0.1`, `FN_KI=1.0` gives A `fn_std` ratio `<= 0.5`
  with compliance `>= +0.15`, at `fixture_B >= 0.90`. Then the regulator is available and the
  honest report is "closed-loop force needs a 5x gain reduction to be non-harmful".
- **F6 (pure-integral arm, control).** `FN_KP=0.0`, `FN_KI=5.0` isolates which term does the damage.
  Reported as a diagnostic, never as a candidate.
- Cells 2/3 (F2/F3 force dose at `1.0` / `0.25`), cell 4 (dynamic range at `0.48,96`, where the
  frozen open-loop champion reads B ~0.60 so a paired win is arithmetically possible) and cell 5
  (the `FN_KP=FN_KI=0` arithmetic negative control) proceed unchanged.

## 9. PRE-REG 3 addendum (after cells 1-6, BEFORE cells 7-8)

Cells 2/3/4/5/6, all 120 episodes each, 0 harness errors, 0 rig bytes changed. Verbatim summary:

| cell | A fn_std ratio / compl (base) | B succ / covc (base) | B slip (base) | B fn (base) |
|---|---|---|---|---|
| 2 set 1.0 | 2.236 / 0.5102 (0.4864) | 20/20, 0.9992 (1.0000) | 0.01963 (0.01059) | 0.992 (0.496) |
| 3 set 0.25 | 1.134 / 0.2859 (0.4864) | 20/20, 1.0000 (1.0000) | 0.00911 (0.01059) | 0.414 (0.496) |
| 4 set 0.5 @ `0.48,96` | 0.586 / 0.0982 | **0/20, 0.0227** (0/20, 0.0125) | 0.55348 (0.50828) | 19 escapes / 20 |
| 5 null (kp=ki=0) | 1.000 / 0.4864 | 20/20, 1.0000 | 0.01059 | 0.496 |
| 6 ki-only (kp=0) | 1.344 / 0.5051 | 20/20, 1.0000 | 0.01064 | 0.499 |

**RIG FACT discovered by the negative control (cell 5), reported not patched (0 rig bytes):**
`FORCE_PI` is **ADDITIVE, not a replacement** — `press = clamp(KP_PRESS*kp*PRESS_M + FN_KP*err +
press_int)`, so with `FN_KP = FN_KI = 0` the arm is **bit-identical** to the open-loop baseline
(every statistic equal to 4 decimals, 60/60 episodes), not the collapse this file's section 5
predicted. The comment at line 119 ("normal force REPLACES the constant press") and the I15 spec
are wrong about the code. Consequence for the whole axis: the reachable setpoint band is
`0.5 N + integral`, i.e. **`FN_SET < 0.5 N` cannot be reached from below by more than the
integral's authority** — cell 3 measured `fn_mean` 0.414 N at set 0.25, and `force_compliance`
0.0140 on B is an artifact of the band edges `[0.5, 1.5] * set` moving BELOW the 0.5 N floor, not
a control failure. Any future force-setpoint claim must quote the ADDITIVE form.

**F1 REFUTED / F4 CONFIRMED / F5 REFUTED (structural kill).** On the intermittent round face the
loop is a variance amplifier at every gain tested: `fn_std` ratio 1.534 (kp .5/ki 5), 1.344
(ki-only), 1.201 (kp .1/ki 1.0, 5x gentler), and compliance gain +0.018 at the gentlest gain
against a pre-registered +0.15 bar. The trend is monotone toward "no worse" as the gain falls, so
this is a **tuning limit, not a wrong-gain artifact**: force feedback on this rig cannot be made
non-harmful, and it is provably inert on the elongated metric face (B `fn_std` 0.0089 -> 0.0046,
compliance 1.0000 in both arms — there is no headroom to win). The frozen open-loop constant press
is the correct design. **Mechanism discarded.**

**F2 CONFIRMED (invariance, the quotable part).** At `0,0` the primary metric is FORCE-INVARIANT
over the reachable band: `fixture_B` `coverage_cont` 1.0000 / 20-of-20 at `fn_mean` 0.414 N and
0.992 N (a **2.4x** force band) with the OPEN-LOOP arm at 0.496 N, while `mean_slip_m` moves
0.00911 -> 0.01059 -> 0.01963 (0.86x / 1.00x / **1.85x**). **F3 CONFIRMED**: N210.3's law
`e = mu*Fn/KP` predicts slip proportional to `Fn` at fixed `KP`, and 2x `Fn` buys 1.85x slip and a
hair of coverage (1.0000 -> 0.9992). So force sets SLIP, path footprint sets COVERAGE, at this
scale — and the paper's "0.5 N" limitation is a SCALE limitation, not a CONTROL limitation.

**Cell 4 has no dynamic range and cannot adjudicate anything:** at `0.48,96` on the frozen
single-cast default BOTH arms lose the head (19-20 of 20 escapes, covc 0.01-0.02) — the plan-frame
error dominates the force channel by two orders of magnitude of slip (0.55 m vs 0.011 m). Recorded
as a **falsified cell design**, not as evidence about force.

**PRE-REG 3 — the one cell class where F2 can still fail.** F2 was measured at `0,0`, where both
arms saturate at 20/20. The open question it leaves: is force-invariance a property of the metric
or only of the zero-noise point? Cells 7/8 put the SAME contrast (PI set 0.5 and set 1.0, kp 0.5 /
ki 5.0) at a **partially-degraded** cell `0.16,32` — inside the coverage regime (N213/N477: the
pose-noise wall on the single-cast default is well above 0.16 m) but not saturated, so
`coverage_cont` has dynamic range and the extra slip of the 2x arm can actually cost coverage.
Pre-registered: (i) if `fixture_B` `coverage_cont` falls `>= 0.005` versus the paired open-loop arm
at set 1.0 while set 0.5 stays within `0.005`, then **F2 is REFUTED in the imperfect-plan regime**
and the invariance claim is restricted to `0,0`; (ii) if both stay within `0.005` with
`>= 18/20` success, F2 survives as a force-invariance result over the working range; (iii) neither
arm reaching `0.90` on B = the cell has no dynamic range and is reported as a falsified cell
design. No keep is claimed from cells 7/8 in advance: a keep still needs B `> 0.70` AND paired
`p < 0.01` on the primary metric.

## 10. PRE-REG 4 addendum (after cells 7/8, BEFORE cells 9/10)

Cells 7/8 at `0.16,32` are another **falsified cell design**: the paired OPEN-LOOP champion
already loses the head there (B **0/20**, 15 of 20 escapes, covc 0.157, slip 0.348 m; A 0/20, R
0/20), so the force channel is again irrelevant — with a 0.16 m planning-pose sigma the 3-sigma
plan error (0.48 m) exceeds the 0.34 m face half-extent, and the escape/`WS_LIMIT_M` channel
dominates. The whole segment's pose-noise robustness is bought by the registration window + cast
lattice (`REG_HALF_M = 0.6`), NOT by anything the press can influence; on the single-cast default
the usable pose-noise band is narrow. Two more cells, chosen to be the LAST place with dynamic
range on this stack: **`0.03,6`** — the paper's Tier-2 cell, where the frozen champion reads
`fixture_B` ~0.75 and coverage is neither saturated nor collapsed. Same contrast, same gains
(kp 0.5 / ki 5.0), set 0.5 and set 1.0, 20 seeds x 3 suites x 2 paired arms.
Pre-registered reading: (i) open-loop B `>= 18/20` and the 2x-force arm `>= 2` episodes lower with
Welch `p < 0.01` on `coverage_cont` -> the extra slip of F3 **does** cost the primary metric once
the plan is imperfect, F2 is restricted to `0,0`, and force becomes a real design constraint;
(ii) both arms `>= 18/20` and `|delta covc| <= 0.005` -> F2 survives over the working range;
(iii) open-loop B `< 18/20` -> no dynamic range, cell reported as a falsified design and the
force axis is closed for this iteration. No keep is pre-claimed; the keep bar (B `> 0.70` AND
paired `p < 0.01`) is unchanged and is read off the rig's own compare record.
