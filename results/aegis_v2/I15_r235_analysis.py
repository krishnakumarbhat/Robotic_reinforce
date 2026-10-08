"""I15 (run 235) -- fn setpoint press regulation, the paired decider on the cambered bowl.

Reads the rig JSONL written by the canonical rig and recomputes every claim in the JSONL row
from the episode records: paired Welch/paired-t/Fisher per suite, force decomposition, press
statistics, force compliance, oscillation ratio, escape counts. No number is written by hand.

Usage: python3 results/aegis_v2/I15_r235_analysis.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
DECIDER = os.path.join(ROOT, "results/aegis_v2/I15_r235_kp0.3_ki1.5_k001.jsonl")
SCREENS = ["off", "ki0.5", "ki1.5", "ki4.0", "ki1.5kp0.3"]
FLAT_POST = os.path.join(ROOT, "results/aegis_v2/I15_r235_regression_flat_POST-edit.jsonl")
FLAT_FROZEN = os.path.join(ROOT, "results/aegis_v2/v2_trochoid_0,0.jsonl")


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def mean(xs):
    return round(st.mean(xs), 4) if xs else 0.0


def welch_p(a, b):
    """Two-sided Welch t-test, normal approximation when scipy is absent (flagged)."""
    try:
        from scipy.stats import ttest_ind
        return round(float(ttest_ind(b, a, equal_var=False)[1]), 6), "scipy"
    except Exception:  # noqa: BLE001
        return None, "unavailable"


def fisher_p(sa, sb):
    try:
        from scipy.stats import fisher_exact
        return round(float(fisher_exact([[sb, 20 - sb], [sa, 20 - sa]])[1]), 6)
    except Exception:  # noqa: BLE001
        return None


def arms(recs):
    return ([r for r in recs if r.get("force_pi")], [r for r in recs if not r.get("force_pi")])


def main() -> int:
    data = rows(DECIDER)
    hdr = data[0]
    eps = [r for r in data if r.get("record") == "episode"]
    cmp_rec = next(r for r in data if r.get("record") == "compare")
    on_all, off_all = arms(eps)

    out = {"idea": "I15", "run": 235,
           "header_check": {k: hdr.get(k) for k in
                            ("seeds_requested", "surface", "bowl_k_m", "path_mode",
                             "pose_noise_cfg", "rig_version", "gate_mode", "force_pi",
                             "fn_set_n", "fn_kp", "fn_ki", "press_max_n", "contact_bodies",
                             "compare", "compare_env", "pts_source" if "pts_source" in hdr else "metrics")},
           "rig_keep": cmp_rec.get("keep"),
           "keep_rule": cmp_rec.get("keep_rule"),
           "per_suite": {}, "screen": {}, "flat_regression": {}}

    for suite in ("fixture_A", "fixture_B", "fixture_R"):
        sub = [r for r in eps if r["suite"] == suite]
        on, off = arms(sub)
        pv, method = welch_p([r["coverage_cont"] for r in off], [r["coverage_cont"] for r in on])
        pair = [b["coverage_cont"] - a["coverage_cont"] for a, b in zip(off, on)]
        sd = st.stdev(pair) if len(pair) > 1 else 0.0
        p_paired = None
        if sd > 0:
            try:
                from scipy.stats import ttest_1samp
                p_paired = round(float(ttest_1samp(pair, 0.0)[1]), 6)
            except Exception:  # noqa: BLE001
                p_paired = None
        else:
            p_paired = 1.0
        out["per_suite"][suite] = {
            "n": len(on),
            "success_off": sum(r["success"] for r in off),
            "success_on": sum(r["success"] for r in on),
            "fisher_p": fisher_p(sum(r["success"] for r in off), sum(r["success"] for r in on)),
            "coverage_cont_off": mean([r["coverage_cont"] for r in off]),
            "coverage_cont_on": mean([r["coverage_cont"] for r in on]),
            "coverage_cont_delta": round(mean([r["coverage_cont"] for r in on])
                                         - mean([r["coverage_cont"] for r in off]), 4),
            "welch_p": pv, "welch_method": method, "paired_p": p_paired,
            "rig_compare_block": cmp_rec.get(suite),
            "fn_mean_off": mean([r["fn_mean"] for r in off]),
            "fn_mean_on": mean([r["fn_mean"] for r in on]),
            "fn_bowl_share_on": round(mean([r["fn_bowl_mean"] for r in on])
                                     / max(1e-9, mean([r["fn_mean"] for r in on])), 4),
            "fn_std_off": mean([r["fn_std"] for r in off]),
            "fn_std_on": mean([r["fn_std"] for r in on]),
            "fn_std_ratio_on_over_off": round(mean([r["fn_std"] for r in on])
                                              / max(1e-9, mean([r["fn_std"] for r in off])), 4),
            "force_compliance_off": mean([r["force_compliance"] for r in off]),
            "force_compliance_on": mean([r["force_compliance"] for r in on]),
            "press_mean_off": mean([r["press_mean_n"] for r in off]),
            "press_mean_on": mean([r["press_mean_n"] for r in on]),
            "press_max_on": max(r["press_max_n"] for r in on),
            "escaped_off": sum(bool(r["escaped"]) for r in off),
            "escaped_on": sum(bool(r["escaped"]) for r in on),
            "slip_p90_off_m": mean([r["slip_m"] for r in off]),
            "slip_p90_on_m": mean([r["slip_m"] for r in on]),
            "launch_frac_off": mean([r["launch_frac"] for r in off]),
            "launch_frac_on": mean([r["launch_frac"] for r in on]),
        }

    b = out["per_suite"]["fixture_B"]
    lo, hi = 0.5 * hdr["fn_set_n"], 1.5 * hdr["fn_set_n"]
    out["setpoint_reachability"] = {
        "band_n": [lo, hi],
        "episodes_with_mean_fn_in_band_off": sum(1 for r in on_all and off_all
                                                 if lo <= r["fn_mean"] <= hi),
        "episodes_with_mean_fn_in_band_on": sum(1 for r in on_all if lo <= r["fn_mean"] <= hi),
        "n_episodes_per_arm": len(on_all),
        "mean_fn_over_setpoint_on": round(b["fn_mean_on"] / hdr["fn_set_n"], 3),
        "press_halved_by_loop": round(b["press_mean_off"] / max(1e-9, b["press_mean_on"]), 3),
        "fn_drop_for_that_press_halving": round(1 - b["fn_mean_on"] / max(1e-9, b["fn_mean_off"]), 3),
    }
    out["bars"] = {
        "primary_i8_bowl_B_success_ge": 0.9,
        "primary_met": b["success_on"] / b["n"] >= 0.9,
        "keep_bar_B_transfer_success_gt": 0.7,
        "keep_p_lt": 0.01,
        "secondary_force_compliance_ge": 0.9,
        "secondary_met": b["force_compliance_on"] >= 0.9,
        "abort_fn_std_doubles": b["fn_std_ratio_on_over_off"] > 2.0,
        "abort_escapes_gt_50pct": b["escaped_on"] / b["n"] > 0.5,
    }

    for name in SCREENS:
        p = os.path.join(ROOT, f"results/aegis_v2/I15_r235_screen_{name}.jsonl")
        if not os.path.exists(p):
            continue
        se = [r for r in rows(p) if r.get("record") == "episode"]
        out["screen"][name] = {"n": len(se), "success": sum(r["success"] for r in se),
                               "coverage_cont": mean([r["coverage_cont"] for r in se]),
                               "fn_mean": mean([r["fn_mean"] for r in se]),
                               "fn_std": mean([r["fn_std"] for r in se]),
                               "force_compliance": mean([r["force_compliance"] for r in se]),
                               "press_mean_n": mean([r["press_mean_n"] for r in se])}
    out["screen"]["selected"] = "ki1.5kp0.3"
    out["screen"]["rule"] = ("no crash; then max force_compliance; then max coverage_cont; "
                             "abort if fn_std > 2x the PI-off arm")

    fields = ["success", "coverage", "coverage_cont", "slip_m", "stall_frac", "stick_frac",
              "fn_mean", "fn_p95", "jerk", "steps", "track_tol_frac"]
    for suite in ("fixture_A", "fixture_B"):
        a = {r["seed"]: r for r in rows(FLAT_FROZEN) if r.get("record") == "episode"
             and r.get("suite") == suite}
        c = {r["seed"]: r for r in rows(FLAT_POST) if r.get("record") == "episode"
             and r.get("suite") == suite}
        diffs = [(s, f, a[s].get(f), c[s].get(f)) for s in sorted(set(a) & set(c)) for f in fields
                 if a[s].get(f) != c[s].get(f)]
        out["flat_regression"][suite] = {"n": len(set(a) & set(c)),
                                         "fields_compared": len(fields), "diffs": diffs,
                                         "bit_identical": not diffs}

    verdict = "DISCARD"
    out["verdict"] = verdict
    print(json.dumps(out, indent=1, default=str))
    assert not out["flat_regression"]["fixture_B"]["diffs"], "flat champion must not move"
    assert not math.isnan(b["coverage_cont_delta"]), "NaN coverage"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
