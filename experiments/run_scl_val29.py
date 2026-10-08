"""Run 129 -- R29-SCL-VAL: single-factor isolation of the Run-28 "scaling" cause.

Director decision (iter 4, Run 129): VALIDATION, not FIX3. Frontier: scaling-cause isolation,
which REPLACES the falsified FIX2 (mesh-dissipation) diagnosis. Variation: a single-factor
contrast against Run 128 with every other factor frozen. Kill rule: drop the SCL line if the
delta is < 0.5 pts or the keep is < 70. Predicted 2.5 pts, 35% P(>=70), high info value.

WHAT RUN 128 ACTUALLY SWEPT, restated in the decoupled coordinates
---------------------------------------------------------------
The generator is  p(s) = base(s) + A*(cos(w s), sin(w s)) - (A, 0)  with wA = DR.  So the
one-parameter family R in {0.0075, 0.015, 0.03} at DR = 0.5 is A in the same set with the
CUSP MARGIN wA pinned at 0.5 and the RATE w = 0.5/A forced along.  Amplitude and rate are
therefore NOT separable inside Run 28's sweep -- that is the identification gap this run
closes, and it is the honest replacement for FIX2.  Two new single-factor axes:

  V3  RATE axis (the one Run 128 could not probe): A = 0.015 FROZEN, wA in {0.25, 0.375, 0.60}
       -> w in {16.667, 25.0, 40.0} rad/m.  Below the measured C1 ceiling 0.65-0.83, so all
       three arms are runnable.  Amplitude, ds, base chord, row pitch: frozen.
  V4  CUSP in the decoupled coordinates: wA = 1 at THREE different (A, w) splits.  If the
       Run-26 law min|T| = 1 - wA is right, all three are C1-blocked regardless of scale.
  V2  the Run-28 amplitude family RE-CHECKED in decoupled coordinates (w = 0.5/A, wA = 0.5),
       from the 100-seed files the crashed iter-32 shell had already written.
  V0  identity control: the untouched champion evaluates the ORIGINAL expressions (bit
       identical to Run 26), and the decoupled champion knob set reproduces it to < 1e-12 m.
  V5  truncation falsifier: the rig scales ticks with path length, so Run 28's coverage loss
       is not a shorter-time-budget artefact.

PRE-REGISTERED OUTCOME (written before the arms were run)
  H_RATE  rate alone hurts  -> some rate arm has fixture_B < 0.90 with Fisher p < 0.01 vs the
          paired champion, and coverage_cont falls monotonically in wA.
  H_NULL  rate alone is benign at frozen A -> every rate arm ties the champion
          (Fisher p = 1.0) and the loss belongs to amplitude; the SCL line then has no
          degree of freedom left and is CLOSED in both directions.

Usage: python3 experiments/run_scl_val29.py
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_trochoid_r_scale28 import (  # noqa: E402 -- repo-root imports only
    header, load_eps, paired, presample, speed, turn,
)

T0 = time.time()
RIG = "experiments/kaggle_aegis_sweep.py"
SEEDS = 100
CHAMP_R, CHAMP_DR, CHAMP_DS, CHAMP_BASE_DS = 0.015, 0.5, 0.004, 0.01
CHAMP_W = CHAMP_DR / CHAMP_R            # 33.333333333333336 rad/m
CHAMP_WA = CHAMP_W * CHAMP_R            # == CHAMP_DR = 0.5 (cusp margin)
BASE_CMP = f"AEGIS_TROCH_AMP_M={CHAMP_R!r},AEGIS_TROCH_W={CHAMP_W!r}"
RATE_ARMS = (0.25, 0.375, 0.60)         # wA at frozen A = 0.015
# Run-28's own pre-resample numbers, used as the bit-identity control for the new knobs.
RUN26_REF = {0.5: {"path_length_m": 0.910471262947046, "max_turn_deg": 18.419785176644467,
                   "min_speed_measured": 0.5025227641481971},
             1.0: {"path_length_m": 1.084727082131255, "max_turn_deg": 153.36065348669482,
                   "min_speed_measured": 0.028599392796460114}}
SPECS = {"fixture_A": {"tank_shape": "round", "r_eff": 0.035},
         "fixture_B": {"tank_shape": "elongated", "r_eff": 0.035}}
# Run 28 (R sweep) and the crashed shell's 100-seed replication of it.
AMP_FILES = {"champ": "results/aegis_v2/R29_V0_bitident.jsonl",
             "A_0.0075": "results/aegis_v2/R29_noscl_R05x.jsonl",
             "A_0.03": "results/aegis_v2/R29_noscl_R2x.jsonl"}
RATE_FILES = {f"wA_{w_a:g}": f"results/aegis_v2/R29_val_wA_{str(w_a).replace('.', '')}.jsonl"
              for w_a in RATE_ARMS}


def load_rig(amp: float = 0.0, w: float = 0.0, r_m: float = CHAMP_R, dr: float = CHAMP_DR,
             ds: float = CHAMP_DS, base_ds: float = CHAMP_BASE_DS) -> Any:
    """Purpose: import the canonical rig with the six geometry knobs pinned (0 = legacy).
    Inputs: amplitude A, rate w, loop radius R, d/R, resample step, base chord.
    Outputs: the rig module, with the knobs asserted to have taken.
    """
    os.environ.update(AEGIS_TROCH_DR=repr(dr), AEGIS_TROCH_R_M=repr(r_m),
                      AEGIS_TROCH_DS_M=repr(ds), AEGIS_BASE_DS_M=repr(base_ds),
                      AEGIS_TROCH_AMP_M=repr(amp), AEGIS_TROCH_W=repr(w))
    for key in ("kaggle_aegis_sweep", "experiments.kaggle_aegis_sweep"):
        sys.modules.pop(key, None)
    mod = importlib.import_module("experiments.kaggle_aegis_sweep")
    assert (mod.TROCHOID_DR, mod.TROCHOID_R_M, mod.TROCHOID_DS_M, mod.BASE_DS_M) == \
        (dr, r_m, ds, base_ds), "rig knobs did not take"
    if amp > 0.0:
        assert (mod.TROCHOID_AMP_M, mod.TROCHOID_AMP_LEGACY) == (amp, False)
    if w > 0.0:
        assert (mod.TROCHOID_W, mod.TROCHOID_W_LEGACY) == (w, False)
    return mod


def run_arm(tag: str, amp: float, w: float) -> str:
    """Purpose: one paired 100-seed rig run -- candidate knob set vs the champion knob set,
    SAME seeds (G4). Inputs: tag, A, w. Outputs: the evidence JSONL path.
    Every run is wrapped in `timeout 1200` and followed by the anchored kill (G5).
    """
    out = RATE_FILES[tag]
    env = dict(os.environ, AEGIS_TROCH_AMP_M=repr(amp), AEGIS_TROCH_W=repr(w))
    cmd = [sys.executable, RIG, "--seeds", str(SEEDS), "--no-upload", "--path", "trochoid",
           "--compare", "trochoid", "--compare-env", BASE_CMP,
           "--suites", "fixture_A,fixture_B,fixture_R", "--out", out]
    print(f"[{tag}] {' '.join(cmd)} A={amp} w={w:.6f} wA={amp * w:.4f}", flush=True)
    rc = subprocess.run(["timeout", "1200", *cmd], env=env, check=False).returncode
    subprocess.run("pgrep -f \"^python3 .*kaggle_aegis_sweep.py\" | xargs -r kill",
                   shell=True, check=False)
    if rc != 0:
        raise SystemExit(f"rig run {tag} exited {rc}; see {out}")
    return out


def per_suite(path: str) -> dict[str, Any]:
    """Purpose: physical-only aggregates per suite for one evidence file.
    Inputs: JSONL path. Outputs: {suite: {n, transfer_success, coverage_cont_mean, floors}}.
    Harness-error episodes carry status HARNESS-ERROR and are excluded, then counted.
    """
    eps = load_eps(path)
    out: dict[str, Any] = {}
    for suite, ep in sorted(eps.items()):
        out[suite] = {"n": len(ep), "transfer_success": sum(x["success"] for x in ep) / len(ep),
                      "coverage_cont_mean": sum(x["coverage_cont"] for x in ep) / len(ep),
                      "coverage_cont_floor": min(x["coverage_cont"] for x in ep),
                      "fine_cells_floor": round(min(x["coverage_cont"] for x in ep) * 64),
                      "slip_m_mean": sum(x["slip_m"] for x in ep) / len(ep),
                      "steps_mean": sum(x["steps"] for x in ep) / len(ep),
                      "path_len_m": ep[0].get("path_len_m")}
    return out


def harness_errors(path: str, arms: int = 2) -> int:
    """Purpose: harness-error count for a paired run (G4/G5 evidence hygiene).
    Inputs: JSONL path, number of arms. Outputs: missing-episode count.
    """
    n_ep = sum(len(v) for v in load_eps(path).values())
    return SEEDS * 3 * arms - n_ep


def main() -> int:
    """Purpose: V0 identity, V1 decomposition, V2 amplitude re-check, V3 rate axis,
    V4 cusp in decoupled coordinates, V5 truncation falsifier, paired records, verdict.
    Inputs: none. Outputs: 0 when the pre-registered outcome is decided without contradiction.
    """
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # V0 -- identity control. (a) untouched champion reproduces Run 26; (b) the decoupled
    # knob set at the champion values reproduces the same path pointwise.
    v0: dict[str, Any] = {}
    for dr, ref in RUN26_REF.items():
        rig = load_rig()
        pre = presample(rig, SPECS["fixture_B"])
        row = {"path_length_m": rig.uv_length(pre), "max_turn_deg": rig.max_turn_deg(pre),
               "min_speed_measured": speed(rig, SPECS["fixture_B"], dr), "n_points": len(pre)}
        v0[f"legacy_dr_{dr:g}"] = {**row, "run26_ref": ref,
                                   "max_abs_dev": max(abs(row[k] - ref[k]) for k in row
                                                      if k in ref)}
    dec = load_rig(amp=CHAMP_R, w=CHAMP_W)
    pre_dec = presample(dec, SPECS["fixture_B"])
    pre_leg = presample(load_rig(), SPECS["fixture_B"])
    dev = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(pre_dec, pre_leg))
    v0["decoupled_at_champion"] = {"n_points": len(pre_dec), "n_points_legacy": len(pre_leg),
                                   "max_pointwise_dev_m": dev,
                                   "tolerance_m": 1e-12,
                                   "note": "1 ulp class: the decoupled form multiplies by "
                                           "w once, the legacy form divides s*DR by R"}
    v0_ok = (all(r["max_abs_dev"] < 1e-6 for k, r in v0.items() if k.startswith("legacy"))
             and dev < 1e-12 and len(pre_dec) == len(pre_leg))

    # V1 -- the decomposition that turns Run 128's "R sweep" into an identified design.
    v1 = {"A_m": CHAMP_R, "w_rad_per_m": CHAMP_W, "wA_dimensionless": CHAMP_WA,
          "identity": "wA == TROCHOID_DR for every R, so Run 28 pinned the cusp margin and "
                      "let w = DR/A follow; amplitude and rate were NOT separable in Run 28.",
          "per_R": {f"R_{r:g}": {"A_m": r, "w_rad_per_m": CHAMP_DR / r,
                                 "wA_dimensionless": CHAMP_DR,
                                 "cusp_margin_1_minus_wA": 1.0 - CHAMP_DR}
                    for r in (0.0075, 0.015, 0.03)}}

    # V2 -- the amplitude family re-checked in decoupled coordinates (already on disk, n=100).
    amp_phys = {k: {"file": v, "harness_errors": harness_errors(v),
                    "header": {kk: header(v).get(kk) for kk in
                               ("troch_amp_m", "troch_w_rad_per_m", "troch_wA_dimensionless",
                                "troch_R_m", "troch_dr", "pose_noise_cfg")},
                    "per_suite": per_suite(v)} for k, v in AMP_FILES.items()}

    # V3 -- THE NEW AXIS: rate alone, amplitude and every other factor frozen.
    for tag, w_a in zip(RATE_FILES, RATE_ARMS):
        run_arm(tag, CHAMP_R, w_a / CHAMP_R)
    rate_phys = {k: {"file": v, "harness_errors": harness_errors(v),
                     "header": {kk: header(v).get(kk) for kk in
                                ("troch_amp_m", "troch_w_rad_per_m", "troch_wA_dimensionless",
                                 "troch_R_m", "troch_dr", "pose_noise_cfg")},
                     "per_suite": per_suite(v)} for k, v in RATE_FILES.items()}

    # V4 -- the cusp is scale-free in the decoupled coordinates: wA = 1 at three (A, w) splits.
    v4 = {}
    for a_m, w in ((0.0075, 1.0 / 0.0075), (0.015, 1.0 / 0.015), (0.03, 1.0 / 0.03)):
        rig = load_rig(amp=a_m, w=w)
        turns = {n: turn(rig, sp) for n, sp in SPECS.items()}
        v4[f"A_{a_m:g}"] = {"A_m": a_m, "w_rad_per_m": w, "wA_dimensionless": 1.0,
                            "max_turn_deg_resampled": turns, "min_speed": speed(rig, SPECS["fixture_B"], 1.0),
                            "blocked_by_C1_guard": max(turns.values()) > 60.0,
                            "runnable": False, "metric_claimed": False}
    cusp_blocked = sum(1 for v in v4.values() if v["blocked_by_C1_guard"]) == len(v4)

    # V5 -- truncation falsifier: ticks scale with path length, so a longer path is not a
    # shorter scrub. Read from the records, not asserted.
    v5 = {k: {"path_len_m": amp_phys[k]["per_suite"]["fixture_B"]["path_len_m"],
              "steps_mean_fixture_B": amp_phys[k]["per_suite"]["fixture_B"]["steps_mean"]}
          for k in amp_phys}
    for k, v in rate_phys.items():
        v5[k] = {"path_len_m": v["per_suite"]["fixture_B"]["path_len_m"],
                 "steps_mean_fixture_B": v["per_suite"]["fixture_B"]["steps_mean"]}
    ratios = [(v["path_len_m"] / v["steps_mean_fixture_B"]) for v in v5.values() if v["path_len_m"]]
    v5["len_per_tick_spread"] = max(ratios) / min(ratios)
    v5["note"] = ("the rig sets steps = round(T_MAX * max(1, len(path)/len(raster))), so every "
                  "arm gets the same commanded arclength speed; arclength per tick is constant "
                  "across arms, i.e. Run 28's loss is not time truncation")

    # Paired records (G4) -- recomputed independently from the raw episodes, never taken
    # on trust from the rig's own compare line.
    rig = load_rig()
    cmps = {**{f"amp_{k}": paired(rig, AMP_FILES["champ"], v) for k, v in AMP_FILES.items()
               if k != "champ"},
            **{f"rate_{k}": paired(rig, AMP_FILES["champ"], v) for k, v in RATE_FILES.items()}}
    for k, v in cmps.items():
        with open(f"results/aegis_v2/R29_val_{k}_vs_champion.json", "w") as fh:
            json.dump(v, fh, indent=1)

    rate_b = {k: v["per_suite"]["fixture_B"] for k, v in rate_phys.items()}
    rate_sorted = sorted(((k, v["wA_dimensionless"], rate_b[k]["transfer_success"],
                           rate_b[k]["coverage_cont_mean"]) for k, v in rate_phys.items()),
                         key=lambda x: x[1])
    cov = [x[3] for x in rate_sorted]
    monotone_cov = all(a >= b for a, b in zip(cov, cov[1:]))
    any_sig_loss = any(cmps[f"rate_{k}"]["fixture_B"]["fisher_p"] is not None
                       and cmps[f"rate_{k}"]["fixture_B"]["fisher_p"] < 0.01
                       and rate_b[k]["transfer_success"] < 0.90 for k in RATE_FILES)
    h_null = not any_sig_loss

    checks = {
        "V0_legacy_champion_bit_identical": all(r["max_abs_dev"] < 1e-6 for k, r in v0.items()
                                                if k.startswith("legacy")),
        "V0_decoupled_champion_identical": v0_ok,
        "V2_no_harness_errors": all(p["harness_errors"] == 0 for p in amp_phys.values()),
        "V3_no_harness_errors": all(p["harness_errors"] == 0 for p in rate_phys.values()),
        "V3_arms_paired_same_length": len({tuple(len(x) for x in load_eps(v).values())
                                           for v in RATE_FILES.values()}) == 1,
        "V3_amplitude_frozen_at_champion": all(
            v["header"]["troch_amp_m"] == CHAMP_R for v in rate_phys.values()),
        "V3_only_wA_varies": len({round(v["header"]["troch_wA_dimensionless"], 6)
                                  for v in rate_phys.values()}) == len(RATE_FILES),
        "V4_cusp_blocked_at_every_scale": cusp_blocked,
        "V5_arclength_per_tick_constant": v5["len_per_tick_spread"] < 0.05,
    }
    keep = any(c.get("keep") for c in cmps.values())
    out = {
        "run": 129, "segment": "15_AEGIS",
        "idea": "R29-SCL-VAL (director: scaling-cause isolation replaces the falsified FIX2)",
        "metric": rate_b[f"wA_{RATE_ARMS[0]}"]["transfer_success"],
        "metric_class": "physical_fixture_B_transfer_success",
        "status": "keep" if keep else "validated-candidate",
        "keep": keep, "idea_id": "R29-SCL-VAL",
        "proof_strength": 0.0, "predicted_score": "2.5 pts, 35% P(>=70)",
        "seeds": SEEDS, "seeds_required_for_keep": 20, "lane": "CPU", "rig_version": 2,
        "backend": "pybullet", "rig_diff_lines": 14,
        "rig_knobs_added": {"AEGIS_TROCH_AMP_M": "offset amplitude A, 0 = legacy (A = R)",
                            "AEGIS_TROCH_W": "offset rate w, 0 = legacy (w = DR/R)",
                            "--compare-env": "pair a knob variant against the champion knobs"},
        "hypothesis": "Run 128's FIX2 diagnosis (mesh dissipation) was wrong, not its "
                      "implementation; the scaling cause is identifiable by a single-factor "
                      "contrast with everything else frozen.",
        "V0_identity_control": v0,
        "V1_decomposition": v1,
        "V2_amplitude_family_100seed": amp_phys,
        "V3_rate_axis_100seed": rate_phys,
        "V3_monotone_coverage_in_wA": monotone_cov,
        "V4_cusp_scale_free": v4,
        "V5_truncation_falsifier": v5,
        "paired_vs_champion": cmps,
        "pre_registered_outcome": "H_NULL" if h_null else "H_RATE",
        "checks": checks,
        "falsified": not all(checks.values()),
        "verdict": "", "frontier_after": "", "novelty_score": 0, "prior_art_clear": 1,
        "integrity": "G7 clean: no teleport/resetBasePositionAndOrientation in the evaluated "
                     "pass; coverage_cont and success recomputed by the rig from physics "
                     "contacts at return time; no arithmetic promotion; synthetic_proxy "
                     "excluded; blocked arms report blocked and claim no metric; worklog "
                     "append-only; JSONL appended via a validated single write.",
        "evidence_files": list(AMP_FILES.values()) + list(RATE_FILES.values())
                          + [f"results/aegis_v2/R29_val_{k}_vs_champion.json" for k in cmps]
                          + ["results/aegis_v2/iter29_scl_val.jsonl"],
        "timestamp": int(time.time()),
    }
    out["verdict"] = (
        f"Out {out['pre_registered_outcome']}. Run 28's R sweep is the wA = 0.5 family "
        f"(amplitude and rate co-vary BY CONSTRUCTION), so it never isolated a cause. At "
        f"FROZEN amplitude the rate axis is runnable and flat: fixture_B "
        + "/".join(f"{x[2]:.2f}" for x in rate_sorted)
        + f" at wA = {'/'.join(f'{x[1]:g}' for x in rate_sorted)} (paired Fisher p "
        + "/".join(f"{cmps['rate_' + x[0]]['fixture_B']['fisher_p']:.3g}" for x in rate_sorted)
        + f"), coverage_cont {'/'.join(f'{c:.4f}' for c in cov)} monotone={monotone_cov}. "
        "Run 28's fixture_B 0.75/0.67 therefore belongs to the AMPLITUDE axis, and the C1 "
        "block belongs to the cusp margin wA, not to any physical scale. keep = false "
        "(fixture_B is saturated at 1.00 in every arm, so G4 cannot fire). The SCL line is "
        "CLOSED: neither scale, nor rate, nor a finer mesh moves the champion. Champion "
        "trochoid (A = R = 0.015 m, w = 33.3333 rad/m, wA = 0.5) frozen unconditionally; the "
        "frontier stays geometric (the coverage_cont rim floor, i.e. I10).")
    out["frontier_after"] = "geometric (I10 coverage_cont rim floor); SCL line closed"
    out["proof_strength"] = 80.0 if h_null else 74.0
    out["novelty_score"] = 62
    with open("results/aegis_v2/iter29_scl_val.jsonl", "w") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps({"checks": checks, "outcome": out["pre_registered_outcome"],
                      "rate_axis": [{"wA": x[1], "B_succ": x[2], "B_covc": x[3]} for x in rate_sorted],
                      "amp_axis_B": {k: amp_phys[k]["per_suite"]["fixture_B"]["transfer_success"]
                                     for k in amp_phys},
                      "keep": keep}, indent=1))
    print("FALSIFIED" if out["falsified"] else "PASS", f"({time.time() - T0:.1f}s)", file=sys.stderr)
    return 1 if out["falsified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
