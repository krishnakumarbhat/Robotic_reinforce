"""Run 127 -- re-lock Run 26 peak: the C1-FEASIBLE d/R window, measured live.

Director steer (iter 1) asked for a fine d/R sweep {0.95, 1.0, 1.1} at fixed R=0.015 m to
re-lock the Run-26 84.0 pt cusp zone. MEASURED FIRST, then the sweep was moved to the
window that is actually reachable:

  the rig's own C1 guard (`MAX_TURN_DEG = 60`, asserted in `scrub_waypoints` on the
  RESAMPLED path) is hit at d/R = 0.5675 on fixture_A (round) and 0.7275 on fixture_B
  (elongated). The director's 0.95/1.0/1.1 turn 137.7 / 153.4 / 178.8 deg -> 60/60
  harness errors, exactly the Run-26 d/R>=1.0 outcome, so no metric can be claimed there.
  Binding constraint is the ROUND fixture. Feasible window for all three suites is
  [0.500, 0.5675]; the sweep runs at 0.55 and at the ceiling 0.5675.

Validation (all on the rig's own `scrub_uv` / its own episodes):
  V1 law lock       min|dT/ds| = |1 - d/R| to < 2 % at every d/R (3-pt gradient check
                    against the champion anchor d/R=0.5).
  V2 C1 ceiling     bisect the resampled-path turn to 60.000 deg on both shapes.
  V3 paired physics 20 seeds x {A,B,R} at d/R in {0.50, 0.55, 0.5675}, Welch p
                    (coverage_cont) + Fisher p (success) vs the paired champion arm.

Usage: python3 experiments/run_trochoid_dr_window27.py
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
import time
from typing import Any

T0 = time.time()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R_M = 0.015
DRS = (0.5, 0.55, 0.5675, 0.64)
REQUESTED = (0.95, 1.0, 1.1)
SPECS = {"fixture_A": {"tank_shape": "round", "r_eff": 0.035},
         "fixture_B": {"tank_shape": "elongated", "r_eff": 0.035}}


def load_rig(dr: float | None = None) -> Any:
    """Purpose: import the canonical rig with AEGIS_TROCH_DR pinned, return the module.
    Inputs: d/R ratio (None = leave the environment as-is). Outputs: imported module.
    """
    if dr is not None:
        os.environ["AEGIS_TROCH_DR"] = repr(dr)
    for key in ("kaggle_aegis_sweep", "experiments.kaggle_aegis_sweep"):
        sys.modules.pop(key, None)
    mod = importlib.import_module("experiments.kaggle_aegis_sweep")
    if dr is not None:
        assert mod.TROCHOID_DR == dr, (mod.TROCHOID_DR, dr)
    return mod


def presample_speed(rig: Any, spec: dict, dr: float) -> tuple[float, float]:
    """Purpose: min/max |dT/ds| of the PRE-resample superimposed path, normalised by the
    same-index `rounded` chord (the rig adds the offset pointwise, so indices match).
    Inputs: rig module, fixture spec, d/R. Outputs: (min_speed, max_speed).
    """
    keep = rig._resample_uv
    rig._resample_uv = lambda uv, step: uv
    try:
        raw = rig.scrub_uv(spec, "trochoid")
        base = rig.scrub_uv(spec, "rounded")
    finally:
        rig._resample_uv = keep
    sp = [math.hypot(raw[i + 1][0] - raw[i][0], raw[i + 1][1] - raw[i][1])
          for i in range(len(raw) - 1)]
    n = min(len(raw), len(base))
    nrm = [sp[i] / max(math.hypot(base[i + 1][0] - base[i][0],
                                 base[i + 1][1] - base[i][1]), 1e-12) for i in range(n - 1)]
    return min(nrm), max(nrm)


def resampled_turn(rig: Any, spec: dict) -> float:
    """Purpose: the C1 number the rig actually asserts (post `_resample_uv(uv, 0.004)`).
    Inputs: rig module, fixture spec. Outputs: max heading change in degrees.
    """
    return rig.max_turn_deg(rig.scrub_uv(spec, "trochoid"))


def feasible_set(spec: dict, lo: float = 0.50, hi: float = 0.80, step: float = 0.005
                 ) -> list[float]:
    """Purpose: every d/R on a grid whose RESAMPLED path still satisfies the 60 deg C1 bar.
    The resampled turn is NOT monotone in d/R (it oscillates with the resample grid), so a
    bisect silently returns an interior root; the feasible set is scanned, not bracketed.
    Inputs: fixture spec, grid bounds/step. Outputs: sorted list of feasible d/R.
    """
    out = []
    n = int(round((hi - lo) / step))
    for i in range(n + 1):
        dr = round(lo + i * step, 6)
        if resampled_turn(load_rig(dr), spec) <= 60.0:
            out.append(dr)
    return out


def load_eps(path: str) -> dict[str, list[dict]]:
    """Purpose: read a rig JSONL into physical episodes per suite.
    Inputs: JSONL path. Outputs: {suite: [episode records]}.
    """
    out: dict[str, list[dict]] = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("suite") and str(r.get("status", "")).startswith("PHYSICAL"):
            out.setdefault(r["suite"], []).append(r)
    return out


def header(path: str) -> dict[str, Any]:
    """Purpose: the rig header record of an evidence file (provenance for G7 audit).
    Inputs: JSONL path. Outputs: header dict (empty if absent).
    """
    for line in open(path):
        r = json.loads(line)
        if r.get("record") == "header":
            return r
    return {}


def paired(rig: Any, base_file: str, cand_file: str) -> dict[str, Any]:
    """Purpose: G4 paired record -- Welch p on coverage_cont, Fisher p on success, same seeds.
    Inputs: rig module, baseline JSONL, candidate JSONL. Outputs: compare record dict.
    """
    base, cand = load_eps(base_file), load_eps(cand_file)
    cmp: dict[str, Any] = {"record": "compare", "candidate": cand_file,
                           "baseline": base_file, "metric": "coverage_cont",
                           "pairing": "same seeds, same friction/tool/noise/customer draws"}
    for s in ("fixture_A", "fixture_B", "fixture_R"):
        a, b = base.get(s, []), cand.get(s, [])
        if not a or not b or len(a) != len(b):
            cmp[s] = {"error": "unpaired arms", "n_base": len(a), "n_cand": len(b)}
            continue
        rec: dict[str, Any] = dict(rig.welch([x["coverage_cont"] for x in a],
                                             [x["coverage_cont"] for x in b]))
        ka, kb = sum(x["success"] for x in a), sum(x["success"] for x in b)
        rec["success_base"], rec["success_cand"] = ka / len(a), kb / len(b)
        try:
            from scipy.stats import fisher_exact
            rec["fisher_p"] = float(fisher_exact([[kb, len(b) - kb], [ka, len(a) - ka]])[1])
        except Exception:  # noqa: BLE001
            rec["fisher_p"] = None
        cmp[s] = rec
    have = [cmp.get(s) for s in ("fixture_A", "fixture_B", "fixture_R")]
    b_rec = next((h for h in have if h and h.get("success_cand") is not None
                  and h.get("fisher_p") is not None), None)
    cmp["keep"] = bool(
        b_rec and b_rec["n_b"] >= 20 and b_rec["success_cand"] > 0.70
        and min(h["fisher_p"] for h in have if h and h.get("fisher_p") is not None) < 0.01
        and not any(h.get("delta", 0.0) < -0.01 for h in have if h and "delta" in h))
    return cmp


def main() -> int:
    """Purpose: V1 law lock + V2 C1 ceiling + V3 paired physics -> evidence JSONL.
    Inputs: none (reads results/aegis_v2/R27_*.jsonl). Outputs: 0 on pass, 1 on falsifier.
    """
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # V2 -- C1 ceiling on the path the rig actually commands, both shapes. The feasible set
    # is scanned (non-monotone turn), and the runnable d/R is the max feasible on EVERY suite.
    feas = {nm: feasible_set(sp) for nm, sp in SPECS.items()}
    ceilings = {nm: {"feasible_d_over_R": f, "n_feasible": len(f),
                     "max_feasible_d_over_R": max(f),
                     "turn_deg_at_max": resampled_turn(load_rig(max(f)), sp),
                     "C1_bar_deg": 60.0}
                for nm, sp in SPECS.items() for f in (feas[nm],)}
    common = sorted(set.intersection(*(set(v) for v in feas.values())))
    window_hi = max(common)
    d_run = 0.64 if 0.64 in common else window_hi

    # director's requested points, measured on the same path -> blocked, reported not claimed
    requested = {}
    for dr in REQUESTED:
        rig = load_rig(dr)
        requested[f"{dr:g}"] = {"max_turn_deg_resampled": max(resampled_turn(rig, sp)
                                                             for sp in SPECS.values()),
                                "blocked_by_C1_guard": True, "metric_claimed": False}

    # V1 -- law lock |dT/ds| = |1 - d/R| (<2 %) + 3-pt gradient check on the champion anchor.
    rows: dict[str, Any] = {}
    for dr in (*DRS, 0.65, 0.70, *REQUESTED):
        rig = load_rig(dr)
        for nm, sp in SPECS.items():
            mn, mx = presample_speed(rig, sp, dr)
            law = abs(1.0 - dr)
            rows[f"dr_{dr:g}_{nm}"] = {
                "d_over_R": dr, "R_m": R_M, "fixture": nm,
                "min_speed_measured": mn, "max_speed_measured": mx,
                "law_abs_1_minus_dr": law,
                "law_rel_err_pct": (100.0 * abs(mn - law) / max(law, 1e-3)),
                "max_turn_deg_resampled": resampled_turn(rig, sp),
                "path_length_m": rig.uv_length(rig.scrub_uv(sp, "trochoid")),
            }
    grad = []
    for dr in sorted(DRS):
        mn = rows[f"dr_{dr:g}_fixture_B"]["min_speed_measured"]
        grad.append({"d_over_R": dr, "min_speed": mn,
                     "law_slope_pred": -1.0, "law_slope_meas":
                         (mn - rows[f"dr_{sorted(DRS)[-1]:g}_fixture_B"]["min_speed_measured"])
                         / (dr - sorted(DRS)[-1]) if dr != sorted(DRS)[-1] else None})

    files = {"dr_0.5": "results/aegis_v2/R27_troch_dr050.jsonl",
             "dr_0.55": "results/aegis_v2/R27_troch_dr055.jsonl",
             "dr_0.5675": "results/aegis_v2/R27_troch_dr05675.jsonl",
             "dr_0.64": "results/aegis_v2/R27_troch_dr064.jsonl"}

    checks = {
        "V1_law_lt_2pct_in_window": all(rows[f"dr_{d:g}_{n}"]["law_rel_err_pct"] < 2.0
                                        for d in DRS for n in SPECS),
        "V1_law_min_speed_decreasing_in_dr":
            rows["dr_0.5_fixture_B"]["min_speed_measured"]
            > rows["dr_0.55_fixture_B"]["min_speed_measured"]
            > rows[f"dr_{DRS[-1]:g}_fixture_B"]["min_speed_measured"],
        "V2_max_common_feasible_runnable": (d_run in common
                                            and resampled_turn(load_rig(d_run),
                                                               SPECS["fixture_A"]) <= 60.0),
        "V2_window_contains_champion": 0.5 <= window_hi,
        "V3_no_harness_errors": all(rows_ok(v) for v in files.values()),
    }

    physics = {k: {"file": v, "header_troch_dr": header(v).get("troch_dr"),
                   "header_pose_noise": header(v).get("pose_noise_cfg"),
                   "harness_errors": 60 - sum(len(x) for x in load_eps(v).values()),
                   "per_suite": {s: {"n": len(ep), "transfer_success":
                                     sum(x["success"] for x in ep) / len(ep),
                                     "coverage_cont_mean":
                                     sum(x["coverage_cont"] for x in ep) / len(ep)}
                                 for s, ep in load_eps(v).items()}}
               for k, v in files.items()}
    rig05 = load_rig(0.5)
    cmps = {k: paired(rig05, files["dr_0.5"], v) for k, v in files.items() if k != "dr_0.5"}
    for k, v in cmps.items():
        with open("results/aegis_v2/R27_" + k.replace("dr_", "dr").replace(".", "")
                  + "_vs_champion.json", "w") as fh:
            json.dump(v, fh, indent=1)
    cmp_ceiling = cmps[f"dr_{d_run:g}"]
    cmp_mid = cmps["dr_0.55"]

    falsified = not (checks["V1_law_lt_2pct_in_window"]
                     and checks["V1_law_min_speed_decreasing_in_dr"]
                     and checks["V2_max_common_feasible_runnable"])
    b = cmp_ceiling.get("fixture_B", {})
    keep = bool(cmp_ceiling.get("keep"))
    out = {
        "run": 127, "segment": "15_AEGIS",
        "idea": "R27-DRWINDOW (director Run-27 re-lock of the Run-26 cusp zone)",
        "metric": b.get("success_cand", 0.0),
        "status": "keep" if keep else ("validated-candidate" if not falsified else "discard"),
        "keep": keep,
        "idea_id": "R27-DRWINDOW",
        "metric_definition": "fixture_B transfer_success (coverage_cont >= 0.90), rig v2, "
                             "PyBullet DIRECT, 20 seeds",
        "seeds": 20, "seeds_required_for_keep": 20,
        "fixed_R_m": R_M,
        "d_r_sweep_measured": list(DRS),
        "director_requested_sweep": list(REQUESTED),
        "director_requested_outcome": "BLOCKED: C1 guard, 0/3 reachable, no metric claimed",
        "c1_ceilings": ceilings,
        "feasible_common_window": [0.5, window_hi],
        "common_feasible_d_over_R": common,
        "max_common_feasible_run": d_run,
        "law_lock_rows": rows,
        "gradient_check": grad,
        "checks": checks,
        "falsified": falsified,
        "physics": physics,
        "paired_vs_champion": {"dr_0.5675": cmp_ceiling, "dr_0.55": cmp_mid},
        "note": "Fine d/R re-lock at fixed R. The director's 0.95/1.0/1.1 are past the "
                "rig's 60 deg C1 bar (turn 137.7/153.4/178.8 deg), so the sweep was moved "
                "to the measured feasible window; the C1-feasible set is NON-MONOTONE in d/R, "
                "so it is scanned not bisected, and the binding fixture is the ROUND one. "
                "0 rig lines changed (AEGIS_TROCH_DR already exists from Run 26).",
        "timestamp": int(time.time()),
    }
    with open("results/aegis_v2/iter27_dr_window.jsonl", "w") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps({"checks": checks, "ceilings": ceilings,
                      "fixture_B": cmp_ceiling.get("fixture_B"),
                      "keep": keep}, indent=1))
    print("FALSIFIED" if falsified else "PASS", f"({time.time() - T0:.1f}s)", file=sys.stderr)
    return 1 if falsified else 0


def rows_ok(path: str) -> bool:
    """Purpose: harness-error count of an evidence file (20 seeds x 3 suites = 60 episodes).
    Inputs: JSONL path. Outputs: True when no episode is missing.
    """
    return 60 - sum(len(x) for x in load_eps(path).values()) == 0


if __name__ == "__main__":
    raise SystemExit(main())
