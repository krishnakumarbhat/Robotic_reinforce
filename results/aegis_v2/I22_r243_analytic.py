"""I22 row-centring pre-count -- NO PHYSICS, closed form on the rig's own scoring kernel.

Purpose: price the one-term row-list fix (`rows = v0 + (r + 0.5*CELL_M)` vs
`v0 + r*CELL_M`) BEFORE any physics, using PyBulletScrub._coverage_cont VERBATIM (bound off a
1-field shim, so this is the rig's own kernel, not a re-implementation). Predicts the decider
and the C1 / arclength cost; if the physics lands somewhere else, the prediction is wrong and
so is the mechanism.
Inputs: none (module-level sweep). Outputs: results/aegis_v2/I22_r243_analytic.json.
"""
from __future__ import annotations

import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "experiments"))
os.environ.setdefault("AEGIS_UPLOAD", "0")
os.environ.setdefault("AEGIS_INSTALL", "0")

import kaggle_aegis_sweep as rig  # noqa: E402

R_EFF = {0: 0.035, 1: 0.04, 2: 0.05}          # min(half[0], half[1]) of TOOL_SHAPES
SUITES = {"fixture_B": {"tank_shape": "elongated"}, "fixture_A": {"tank_shape": "round"}}


class _Shim:
    """The rig kernel reads exactly one field off self."""

    def __init__(self, r_eff: float) -> None:
        self.r_eff = r_eff


def ceil_of(uv: list, spec: dict, r_eff: float):
    """Analytic coverage_cont + per-cell hit mask of a (u,v) plan under the rig's own kernel."""
    import numpy as np
    org, nu, nv = rig.scrub_grid(spec)
    cov = rig.PyBulletScrub._coverage_cont(_Shim(r_eff), uv, org, nu, nv)
    fu, fv = 2 * nu, 2 * nv
    cu = org[0] + (np.arange(fu) + 0.5) * rig.FINE_M
    cv = org[1] + (np.arange(fv) + 0.5) * rig.FINE_M
    G = np.stack(np.meshgrid(cu, cv, indexing="ij"), axis=-1).reshape(-1, 2)
    P = np.asarray(uv[::2] if len(uv) > 400 else uv, dtype=np.float64)
    hit = np.zeros(len(G), dtype=bool)
    for k in range(0, len(P), 256):
        d = np.linalg.norm(G[:, None, :] - P[None, k:k + 256, :], axis=-1)
        hit |= (d <= r_eff).any(axis=1)
    return cov, hit.reshape(fu, fv)


def max_turn_deg(uv: list) -> float:
    """C1 assertion metric: largest per-dense-segment heading change, degrees."""
    a = []
    for i in range(1, len(uv)):
        dx, dy = uv[i][0] - uv[i - 1][0], uv[i][1] - uv[i - 1][1]
        if math.hypot(dx, dy) > 1e-12:
            a.append(math.atan2(dy, dx))
    return max(abs(math.degrees((a[i + 1] - a[i] + math.pi) % (2 * math.pi) - math.pi))
               for i in range(len(a) - 1)) if len(a) > 1 else 0.0


def arclen(uv: list) -> float:
    """Path length in metres."""
    return sum(math.hypot(uv[i][0] - uv[i - 1][0], uv[i][1] - uv[i - 1][1]) for i in range(1, len(uv)))


def main() -> int:
    out: dict = {"kernel": "PyBulletScrub._coverage_cont (verbatim)",
                 "law": "rows = v0 + (r + 0.5*ROW_CENTRE)*CELL_M, v0 = -side/2; "
                        "dv* = CELL_M/2 = +0.025 m, independent of side, nv and r_eff",
                 "suites": {}}
    for sname, spec in SUITES.items():
        org, nu, nv = rig.scrub_grid(spec)
        side = 0.12 if spec["tank_shape"] == "elongated" else 0.18
        rec: dict = {"side_m": side, "cell_m": rig.CELL_M, "grid_org": org, "n_u": nu, "n_v": nv,
                     "fine_v_centres": [round(org[1] + (j + 0.5) * rig.FINE_M, 4)
                                        for j in range(2 * nv)]}
        plans = {}
        for tag, rc in (("champion_ROW_CENTRE_0", 0.0), ("i22_ROW_CENTRE_1", 1.0)):
            rig.ROW_CENTRE = rc
            uv = rig.scrub_uv(spec, "trochoid")
            rows = sorted({round(v, 4) for v in
                           [p[1] for p in rig.scrub_uv(spec, "rounded")]})
            cell = {"n_points": len(uv), "path_len_m": round(arclen(uv), 4),
                    "max_turn_deg": round(max_turn_deg(uv), 2),
                    "plan_v_span_m": [round(min(v for _, v in uv), 4), round(max(v for _, v in uv), 4)],
                    "base_rows_v": rows, "ceilings": {}, "missed_v_rows": {}}
            for tid, r_eff in R_EFF.items():
                cov, hit = ceil_of(uv, spec, r_eff)
                cell["ceilings"][f"tool{tid}_r_eff_{r_eff}"] = round(cov, 4)
                cell["missed_v_rows"][f"tool{tid}_r_eff_{r_eff}"] = (
                    [j for j in range(2 * nv) if not hit[:, j].any()])
            plans[tag] = cell
        rec["plans"] = plans
        rec["delta"] = {
            "coverage": {k: round(plans["i22_ROW_CENTRE_1"]["ceilings"][k]
                                  - plans["champion_ROW_CENTRE_0"]["ceilings"][k], 4)
                         for k in plans["champion_ROW_CENTRE_0"]["ceilings"]},
            "path_len_ratio": round(plans["i22_ROW_CENTRE_1"]["path_len_m"]
                                    / plans["champion_ROW_CENTRE_0"]["path_len_m"], 6),
            "max_turn_deg": [plans["champion_ROW_CENTRE_0"]["max_turn_deg"],
                             plans["i22_ROW_CENTRE_1"]["max_turn_deg"]],
            "c1_bound_deg": rig.MAX_TURN_DEG,
        }
        out["suites"][sname] = rec
    rig.ROW_CENTRE = 1.0
    # the same law on the other row modes, to show it is a PLANNER class, not a trochoid quirk
    out["other_modes"] = {}
    for sname, spec in SUITES.items():
        m = {}
        for mode in ("raster", "rounded"):
            rig.ROW_CENTRE = 0.0
            uv0 = rig.scrub_uv(spec, mode)
            rig.ROW_CENTRE = 1.0
            uv1 = rig.scrub_uv(spec, mode)
            m[mode] = {f"tool{t}_r_eff_{r}": {"champion": round(ceil_of(uv0, spec, r)[0], 4),
                                              "i22": round(ceil_of(uv1, spec, r)[0], 4)}
                       for t, r in R_EFF.items()}
        out["other_modes"][sname] = m
    rig.ROW_CENTRE = 1.0
    p = os.path.join(REPO, "results", "aegis_v2", "I22_r243_analytic.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
