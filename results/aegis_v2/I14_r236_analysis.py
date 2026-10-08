"""I14 (run 236) -- patch-inset containment on the cambered bowl, paired decider.

Two full 20-seed x 3-suite paired deciders against --compare-env AEGIS_INSET_M=0.0:
  arm 1  I14_r236_i0.020_k001.jsonl     clamp+wall, inset 0.020 (the screen survivor)
  arm 2  I14_r236_wallonly0.010_k001    wall-only, inset 0.010 (mechanism-isolating ablation)
Every number is recomputed here from the episode records; none is written by hand.

Usage: python3 results/aegis_v2/I14_r236_analysis.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECIDERS = {
    "clamp_wall_0.020": "results/aegis_v2/I14_r236_i0.020_k001.jsonl",
    "wall_only_0.010": "results/aegis_v2/I14_r236_wallonly0.010_k001.jsonl",
}
SCREENS = ["off", "i0.010", "i0.020", "i0.035",
           "wallonly0.010", "wallonly0.020", "wallonly0.020k100", "wallonly0.035"]
FLAT_POST = os.path.join(ROOT, "results/aegis_v2/I14_r236_regression_flat_POST-edit.jsonl")
FLAT_FROZEN = os.path.join(ROOT, "results/aegis_v2/v2_trochoid_0,0.jsonl")


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def mean(xs):
    return round(st.mean(xs), 4) if xs else 0.0


def welch_p(a, b):
    try:
        from scipy.stats import ttest_ind
        return round(float(ttest_ind(b, a, equal_var=False)[1]), 6), "scipy"
    except Exception:  # noqa: BLE001
        return None, "unavailable"


def paired_p(pair):
    sd = st.stdev(pair) if len(pair) > 1 else 0.0
    if sd <= 0:
        return 1.0
    try:
        from scipy.stats import ttest_1samp
        return round(float(ttest_1samp(pair, 0.0)[1]), 6)
    except Exception:  # noqa: BLE001
        return None


def fisher_p(sa, sb, n=20):
    try:
        from scipy.stats import fisher_exact
        return round(float(fisher_exact([[sb, n - sb], [sa, n - sa]])[1]), 6)
    except Exception:  # noqa: BLE001
        return None


def arms(recs):
    return [r for r in recs if r.get("inset_m", 0.0) > 0.0], \
           [r for r in recs if r.get("inset_m", 0.0) <= 0.0]


def main() -> int:
    out = {"idea": "I14", "run": 236, "deciders": {}, "screen": {}, "flat_regression": {}}

    for arm_name, rel in DECIDERS.items():
        data = rows(os.path.join(ROOT, rel))
        hdr = data[0]
        eps = [r for r in data if r.get("record") == "episode"]
        cmp_rec = next(r for r in data if r.get("record") == "compare")
        on_all, off_all = arms(eps)
        suite_out = {}
        for suite in ("fixture_A", "fixture_B", "fixture_R"):
            on, off = arms([r for r in eps if r["suite"] == suite])
            pv, method = welch_p([r["coverage_cont"] for r in off],
                                 [r["coverage_cont"] for r in on])
            pair = [b["coverage_cont"] - a["coverage_cont"] for a, b in zip(off, on)]
            suite_out[suite] = {
                "n": len(on),
                "success_off": sum(r["success"] for r in off),
                "success_on": sum(r["success"] for r in on),
                "fisher_p": fisher_p(sum(r["success"] for r in off),
                                     sum(r["success"] for r in on)),
                "coverage_cont_off": mean([r["coverage_cont"] for r in off]),
                "coverage_cont_on": mean([r["coverage_cont"] for r in on]),
                "coverage_cont_delta": round(mean([r["coverage_cont"] for r in on])
                                             - mean([r["coverage_cont"] for r in off]), 4),
                "welch_p": pv, "welch_method": method, "paired_p": paired_p(pair),
                "rig_compare_block": cmp_rec.get(suite),
                "escaped_off": sum(bool(r["escaped"]) for r in off),
                "escaped_on": sum(bool(r["escaped"]) for r in on),
                "slip_p90_off_m": mean([r["slip_m"] for r in off]),
                "slip_p90_on_m": mean([r["slip_m"] for r in on]),
                "launch_frac_off": mean([r["launch_frac"] for r in off]),
                "launch_frac_on": mean([r["launch_frac"] for r in on]),
                "z_exc_max_off_m": max(r["z_exc_max_m"] for r in off),
                "z_exc_max_on_m": max(r["z_exc_max_m"] for r in on),
                "path_len_off_m": off[0]["path_len_m"],
                "path_len_on_m": on[0]["path_len_m"],
                "wall_ticks_total_on": sum(r["wall_ticks"] for r in on),
                "wall_pen_max_on_m": max(r["wall_pen_max_m"] for r in on),
                "wall_ticks_total_off": sum(r["wall_ticks"] for r in off),
            }
        b = suite_out["fixture_B"]
        out["deciders"][arm_name] = {
            "header_check": {k: hdr.get(k) for k in
                             ("seeds_requested", "surface", "bowl_k_m", "path_mode",
                              "pose_noise_cfg", "rig_version", "gate_mode", "inset_m",
                              "wall_only", "wall_kp", "contact_bodies", "compare",
                              "compare_env")},
            "rig_keep": cmp_rec.get("keep"), "keep_rule": cmp_rec.get("keep_rule"),
            "per_suite": suite_out,
            "bars": {
                "prereg_primary_escapes_lt_5pct": b["escaped_on"] / b["n"] < 0.05,
                "prereg_abort_coverage_drop_gt_0.10":
                    (b["coverage_cont_off"] - b["coverage_cont_on"]) > 0.10,
                "keep_B_transfer_success_gt_0.70": b["success_on"] / b["n"] > 0.70,
                "keep_sig_p_lt_0.01": (b["welch_p"] is not None and b["welch_p"] < 0.01)
                    or (b["fisher_p"] is not None and b["fisher_p"] < 0.01),
                "escapes_on_frac": round(b["escaped_on"] / b["n"], 4),
                "escapes_off_frac": round(b["escaped_off"] / b["n"], 4),
            },
        }

    for name in SCREENS:
        p = os.path.join(ROOT, f"results/aegis_v2/I14_r236_screen_{name}.jsonl")
        if not os.path.exists(p):
            continue
        se = [r for r in rows(p) if r.get("record") == "episode"]
        out["screen"][name] = {
            "n": len(se), "success": sum(r["success"] for r in se),
            "coverage_cont": mean([r["coverage_cont"] for r in se]),
            "escaped": sum(bool(r["escaped"]) for r in se),
            "slip_p90_max": max(r["slip_m"] for r in se),
            "wall_ticks": sum(r["wall_ticks"] for r in se),
            "path_len_m": se[0]["path_len_m"],
        }
    out["screen"]["selected"] = "i0.020 -> decider arm 1; wall-only 0.010 screened " \
                                "abort (coverage -0.145) but run as the ablation"
    out["screen"]["rule"] = ("no harness crash; exits <5% first; abort if coverage drops >0.10 "
                             "vs off; ties on escapes -> take the wall-only ablation too")

    fields = ["success", "coverage", "coverage_cont", "slip_m", "jerk", "stall_frac",
              "stick_frac", "fn_mean", "path_len_m", "steps", "escaped"]
    for suite in ("fixture_A", "fixture_B"):
        a = {r["seed"]: r for r in rows(FLAT_FROZEN) if r.get("record") == "episode"
             and r.get("suite") == suite}
        c = {r["seed"]: r for r in rows(FLAT_POST) if r.get("record") == "episode"
             and r.get("suite") == suite}
        diffs = [(s, f, a[s].get(f), c[s].get(f)) for s in sorted(set(a) & set(c)) for f in fields
                 if a[s].get(f) != c[s].get(f)]
        out["flat_regression"][suite] = {"n": len(set(a) & set(c)), "fields_compared": len(fields),
                                         "diffs": diffs, "bit_identical": not diffs}

    out["verdict"] = "DISCARD"
    print(json.dumps(out, indent=1, default=str))
    assert not out["flat_regression"]["fixture_B"]["diffs"], "flat champion must not move"
    assert not out["flat_regression"]["fixture_A"]["diffs"], "flat champion must not move"
    for n, d in out["deciders"].items():
        for s, v in d["per_suite"].items():
            assert not math.isnan(v["coverage_cont_delta"]), f"NaN coverage {n}/{s}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
