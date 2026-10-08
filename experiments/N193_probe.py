"""N193 attribution probe: is fixture_A's yaw-driven coverage loss PHYSICS or PLAN GEOMETRY?

Purpose: fixture_A (round tank) declares its yaw UNOBSERVABLE, so the plan is built in a
  frame rotated by the full prior yaw error theta, while `coverage_cont` scores a RECTANGULAR
  target patch fixed in the TRUE fixture frame. This probe computes, with NO physics, the
  coverage_cont the metric would report if the head tracked the commanded path PERFECTLY
  (`_coverage_cont` kernel verbatim: fine cells of the true patch, hit iff a contact lies
  within the pad's r_eff). If the measured physics curve equals this curve, the loss is
  plan/score FRAME PAIRING, not friction, slip or contact loss.
Inputs: none (module constants below). Outputs: printed table + results/N193_probe.json.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks"))
import kaggle_aegis_sweep as rig  # noqa: E402

SPEC_A = {"tank_shape": "round", "surface": "glossy", "offset_cm": 0, "angle_deg": 0}
SPEC_B = {"tank_shape": "elongated", "surface": "matte", "offset_cm": 15, "angle_deg": 10}
R_EFF = {0: 0.035, 1: 0.04, 2: 0.05}          # rig TOOL_SHAPES -> min(half[0], half[1])
FINE_M = rig.FINE_M                            # 0.025 m scoring pitch


def target_cells(spec: dict) -> np.ndarray:
    """Fine scoring cells of the patch in the TRUE fixture frame (rig `_coverage_cont`)."""
    org, nu, nv = rig.scrub_grid(spec)
    cu = org[0] + (np.arange(2 * nu) + 0.5) * FINE_M
    cv = org[1] + (np.arange(2 * nv) + 0.5) * FINE_M
    return np.stack(np.meshgrid(cu, cv, indexing="ij"), axis=-1).reshape(-1, 2)


def analytic_coverage(spec: dict, mode: str, theta_deg: float, r_eff: float,
                      dx: float = 0.0, dy: float = 0.0) -> float:
    """coverage_cont of the commanded path under PERFECT tracking, rotated by theta_deg.

    The plan is generated in the (wrong) planned frame exactly as `scrub_waypoints` does --
    `pt(u,v) = origin + R(angle_deg + theta) @ (u,v)` -- and the scoring cells stay in the
    true frame, so a rect plan rotated against a rect target loses the corners.
    """
    uv = rig.scrub_uv(spec, mode)
    yaw = math.radians(float(spec.get("angle_deg", 0))) + math.radians(theta_deg)
    ca, sa = math.cos(yaw), math.sin(yaw)
    P = np.asarray(uv, dtype=np.float64)
    W = np.stack([dx + P[:, 0] * ca - P[:, 1] * sa,
                  dy + P[:, 0] * sa + P[:, 1] * ca], axis=-1)
    if len(W) > 400:                            # the rig subsamples long contact lists
        W = W[::2]
    G = target_cells(spec)
    hit = np.zeros(len(G), dtype=bool)
    for k in range(0, len(W), 256):
        d = np.linalg.norm(G[:, None, :] - W[None, k:k + 256, :], axis=-1)
        hit |= (d <= r_eff).any(axis=1)
    return float(hit.mean())


def main() -> int:
    """Print the analytic yaw curve for both fixtures and the three pads."""
    thetas = [0, 6, 20, 40, 48, 56, 64, 90]
    out = {"thetas_deg": thetas, "r_eff_by_tool": R_EFF, "mode": "trochoid",
           "plan_len_m": {s: round(rig.uv_length(rig.scrub_uv(sp, "trochoid")), 4)
                          for s, sp in (("fixture_A", SPEC_A), ("fixture_B", SPEC_B))},
           "analytic_coverage_cont": {}, "analytic_orbit_coverage_cont": {}}
    for suite, spec in (("fixture_A", SPEC_A), ("fixture_B", SPEC_B)):
        out["analytic_coverage_cont"][suite] = {
            str(t): {str(tid): round(analytic_coverage(spec, "trochoid", t, r), 4)
                     for tid, r in R_EFF.items()} for t in thetas}
        try:
            out["analytic_orbit_coverage_cont"][suite] = {
                str(t): {str(tid): round(analytic_coverage(spec, "orbit", t, r), 4)
                         for tid, r in R_EFF.items()} for t in thetas}
        except (Exception, SystemExit) as exc:  # noqa: BLE001 -- orbit is round-only
            out["analytic_orbit_coverage_cont"][suite] = f"n/a: {type(exc).__name__}: {exc}"
    # A yaw-invariant plan must be flat in theta; report the spread as the falsifiable claim.
    a = out["analytic_coverage_cont"]["fixture_A"]
    spread = max(max(v.values()) - min(v.values()) for v in a.values())
    out["fixture_A_trochoid_max_spread_over_theta"] = round(spread, 4)
    print(json.dumps(out, indent=1))
    print(f"\nfixture_A trochoid coverage_cont spread over theta "
          f"(the yaw cost, perfect tracking): {spread:.4f}")
    with open("results/N193_probe.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
