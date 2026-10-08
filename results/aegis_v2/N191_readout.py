"""Purpose: N191 readout -- compare the two registration estimator arms of a paired run on the
estimator's own error fields, and show that reg_err_yaw_deg is the amount of yaw error REMOVED
while reg_yaw_plan_deg is the plan's residual yaw error.
Inputs: argv[1] = results jsonl. Outputs: stdout table.
"""
import json
import statistics as st
import sys

import numpy as np


def q(xs, p):
    """Purpose: percentile helper. Inputs: list, p. Outputs: value."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    """Purpose: per-suite, per-arm estimator readout. Inputs: none. Outputs: stdout."""
    rows = [json.loads(l) for l in open(sys.argv[1])]
    arms = {}
    for r in rows:
        if r.get("record") != "episode" or r.get("reg_ok") is None:
            continue
        arms.setdefault(r["suite"], {}).setdefault(r["reg_yaw_plan_deg"] is not None, []).append(r)
    # the candidate arm runs first, the paired baseline second, both in file order
    for suite in ("fixture_A", "fixture_B", "fixture_R"):
        eps = [r for r in rows if r.get("record") == "episode" and r["suite"] == suite]
        reg = [r for r in eps if r.get("reg_ok") is not None]
        n = len(reg) // 2
        for i, lab in enumerate(("cand", "base")):
            a = reg[i * n:(i + 1) * n]
            xy = [r["reg_err_xy_m"] for r in a if r["reg_err_xy_m"] == r["reg_err_xy_m"]]
            old = [r["reg_err_yaw_deg"] for r in a if r["reg_err_yaw_deg"] == r["reg_err_yaw_deg"]]
            new = [r["reg_yaw_plan_deg"] for r in a
                   if r.get("reg_yaw_plan_deg") is not None
                   and r["reg_yaw_plan_deg"] == r["reg_yaw_plan_deg"]]
            cov = [r["coverage_cont"] for r in a]
            print(f"{suite:10s} {lab} n={len(a):3d} reg_ok={sum(1 for r in a if r['reg_ok']):3d} "
                  f"succ={sum(r['success'] for r in a):3d} covc={st.mean(cov):.4f} "
                  f"xy med/p90={st.median(xy) * 1000:6.2f}/{q(xy, 0.9) * 1000:6.2f} mm "
                  f"yaw_OLD(removed) med={st.median(old):5.2f} "
                  f"yaw_PLAN(residual) med={st.median(new):5.2f} p90={q(new, 0.9):5.2f} deg")


if __name__ == "__main__":
    main()
