"""N193 readout: measured vs ANALYTIC coverage_cont for fixture_A, per episode.

Purpose: the attribution test. `coverage_cont` is computed by the metric from physics
  contacts; the analytic predictor computes the SAME number from the commanded path alone
  (perfect tracking) with the rig's own `_coverage_cont` kernel. If per-episode measured ==
  analytic, fixture_A's yaw-driven coverage loss is plan/score FRAME PAIRING, not friction,
  slip, contact loss or any force term.
Inputs: results/aegis_v2/N193_*.jsonl. Outputs: results/N193_readout.json + stdout table.
"""
import glob
import importlib.util
import json
import math
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmarks"))
_SPEC = importlib.util.spec_from_file_location("n193p", os.path.join(HERE, "N193_probe.py"))
P = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(P)  # type: ignore[union-attr]

R_EFF = P.R_EFF
FIELDS = ("n", "succ", "covc", "pred", "resid", "maxresid", "r", "yaw_med")


def rows(path: str) -> list:
    """Every jsonl record of one rig file."""
    return [json.loads(l) for l in open(path)]


def stat_block(eps: list) -> dict:
    """Measured vs analytic coverage over one arm's episodes (per-episode pairing)."""
    pred = [P.analytic_coverage(eps[i]["fixture_spec"], _mode(eps[i]), abs(math.degrees(eps[i]["pose_noise"][2])),
                                R_EFF[eps[i]["tool_id"]]) for i in range(len(eps))]
    meas = [r["coverage_cont"] for r in eps]
    pred_s = [a >= 0.90 for a in pred]
    meas_s = [bool(r["success"]) and not r["escaped"] for r in eps]
    agree = sum(a == b for a, b in zip(pred_s, meas_s))
    resid = [m - a for m, a in zip(meas, pred)]
    spread = max(meas) - min(meas)
    r = float(np.corrcoef(meas, pred)[0, 1]) if spread > 1e-12 else None
    return {"n": len(eps),
            "succ": f"{sum(meas_s)}/{len(eps)}",
            "covc": round(st.mean(meas), 4),
            "pred": round(st.mean(pred), 4),
            "resid": round(st.mean(resid), 4),
            "maxresid": round(max(abs(x) for x in resid), 4),
            "r": None if r is None else round(r, 4),
            "analytic_predicts_success": f"{agree}/{len(eps)}",
            "yaw_med": round(st.median(abs(math.degrees(r_["pose_noise"][2])) for r_ in eps), 1)}


def _mode(rec: dict) -> str:
    """The plan the arm ACTUALLY ran: `auto` resolves on the customer shape, so the
    attribution is computed against the plan that produced the contact points."""
    if rec.get("path_mode") != "auto":
        return rec["path_mode"]
    return "orbit" if rec["fixture_spec"].get("tank_shape") == "round" else "trochoid"


def main() -> int:
    """Yaw-only attribution (both plans) + the full-stack paired candidate runs."""
    out: dict = {"yaw_only": {}, "candidate_full_stack": {}, "attribution_auto_s100": {}}
    for path in sorted(glob.glob("results/aegis_v2/N193_yawonly_*.jsonl")):
        base = os.path.basename(path)[len("N193_yawonly_"):-len(".jsonl")]
        recs = rows(path)
        arms: dict = {}
        for r in recs:
            if r.get("record") == "episode":
                arms.setdefault(r["path_mode"], []).append(r)
        out["yaw_only"][base] = {m: stat_block(e) for m, e in arms.items()}
    for path in sorted(glob.glob("results/aegis_v2/N193_orbit_*.jsonl")):
        tag = os.path.basename(path)[len("N193_orbit_"):-len(".jsonl")]
        entry: dict = {}
        for r in rows(path):
            if r.get("record") == "compare":
                entry["compare"] = {s: {k: r[s][k] for k in
                                        ("n_a", "n_b", "mean_a", "mean_b", "delta", "welch_p",
                                         "paired_p", "succ_a", "succ_b", "fisher_p")}
                                    for s in ("fixture_A", "fixture_B") if s in r}
                entry["rig_keep"] = r.get("keep")
                entry["pose_noise_cfg"] = r.get("pose_noise_cfg")
                entry["baseline"] = r.get("baseline")
            if r.get("record") == "summary_mode":
                ps = r.get("per_suite", {})
                entry[r["path_mode"]] = {
                    s: {k: v.get(k) for k in ("transfer_success", "mean_coverage_cont",
                                               "path_len_m", "harness_errors", "escaped_frac",
                                               "mean_slip_m", "mean_stick_frac")}
                    for s, v in ps.items() if s in ("fixture_A", "fixture_B", "fixture_R")}
        out["candidate_full_stack"][tag] = entry
    # N193 attribution on the shipped 100-seed matrix: measured coverage_cont against the
    # ANALYTIC perfect-tracking coverage of the plan the arm actually ran.
    for path in sorted(glob.glob("results/aegis_v2/N193_r304_auto_s100_*.jsonl")):
        tag = os.path.basename(path)[len("N193_r304_auto_s100_"):-len(".jsonl")]
        recs = rows(path)
        arms: dict = {}
        for r in recs:
            if r.get("record") == "episode":
                arms.setdefault((r["suite"], r["path_mode"]), []).append(r)
        out["attribution_auto_s100"][tag] = {
            f"{s}/{m}": stat_block(e) for (s, m), e in sorted(arms.items())}
    print(json.dumps(out, indent=1))
    with open("results/N193_readout.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
