"""I3 mechanism diagnosis -- NO PHYSICS, closed form on the rig's own scoring kernel.

Purpose: price a CONSTANT command offset (du, dv) applied to the champion trochoid plan,
using PyBulletScrub._coverage_cont VERBATIM (bound off a 1-field shim, so this is the rig's
own kernel, not a re-implementation). Attributes the I3 gain to a plan defect or refutes it.
Inputs: none (module-level sweep). Outputs: results/aegis_v2/I3_r242_mechanism.json.
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "experiments"))
os.environ.setdefault("AEGIS_UPLOAD", "0")
os.environ.setdefault("AEGIS_INSTALL", "0")

import kaggle_aegis_sweep as rig  # noqa: E402

R_EFF = {0: 0.035, 1: 0.04, 2: 0.05}          # min(half[0], half[1]) of TOOL_SHAPES
SUITES = {"fixture_B": {"tank_shape": "elongated"}, "fixture_A": {"tank_shape": "round"}}
DVS = [round(-0.05 + 0.005 * k, 3) for k in range(21)]     # -0.050 .. +0.050


class _Shim:
    """The rig kernel reads exactly one field off self."""

    def __init__(self, r_eff: float) -> None:
        self.r_eff = r_eff


def ceil_of(uv: list, spec: dict, r_eff: float) -> tuple:
    """Analytic coverage_cont of a (u,v) plan under the rig's own kernel.
    Returns (coverage, hit_mask as list of [iu, iv])."""
    org, nu, nv = rig.scrub_grid(spec)
    shim = _Shim(r_eff)
    cov = rig.PyBulletScrub._coverage_cont(shim, uv, org, nu, nv)
    import numpy as np
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


def main() -> int:
    out: dict = {"kernel": "PyBulletScrub._coverage_cont (verbatim)", "rows_rule": None,
                 "sweep": {}, "grid": {}}
    for sname, spec in SUITES.items():
        org, nu, nv = rig.scrub_grid(spec)
        side = 0.12 if spec["tank_shape"] == "elongated" else 0.18
        half = 0.20
        rows_champ = [-side * 0.5 + r * rig.CELL_M for r in range(max(1, int(side / rig.CELL_M)))]
        rows_fitted = [-side * 0.5 + side * (k + 0.5) / max(1, int(-(-side // (2 * 0.035))))
                       for k in range(max(1, int(-(-side // (2 * 0.035)))))]
        out["grid"][sname] = {"side_m": side, "cell_m": rig.CELL_M, "org": org, "nu": nu, "nv": nv,
                              "n_fine_cells": 4 * nu * nv,
                              "fine_v_centres": [round(org[1] + (j + 0.5) * rig.FINE_M, 4)
                                                 for j in range(2 * nv)],
                              "champion_rows_v": rows_champ, "fitted_rows_v": rows_fitted}
        if sname == "fixture_B":
            out["rows_rule"] = {
                "champion": f"nv = int({side}/{rig.CELL_M}) = {int(side / rig.CELL_M)}; "
                            f"v0 = -{side}/2 = {-side / 2}; rows = [v0 + r*{rig.CELL_M}] -> "
                            f"{[round(x, 4) for x in rows_champ]} : the band is ANCHORED AT THE "
                            f"-v EDGE and spans only v in "
                            f"[{rows_champ[0]:.3f}, {rows_champ[-1]:.3f}] of a patch whose fine "
                            f"cell centres run to {org[1] + (2 * nv - 0.5) * rig.FINE_M:.4f}",
                "fitted": "n = ceil(side/(2 r_eff)) rows at the band CENTRES, no inset -> spans "
                          f"[-{side / 2}, {side / 2}] exactly",
            }
        base = rig.scrub_uv(spec, "trochoid")
        sw = {}
        for tid, r_eff in R_EFF.items():
            row = {}
            for dv in DVS:
                uv = [(u, v + dv) for (u, v) in base]
                cov, hit = ceil_of(uv, spec, r_eff)
                row[f"{dv:+.3f}"] = round(cov, 4)
            cov0, hit0 = ceil_of(base, spec, r_eff)
            _, hit20 = ceil_of([(u, v + 0.02) for (u, v) in base], spec, r_eff)
            _, hit30 = ceil_of([(u, v + 0.03) for (u, v) in base], spec, r_eff)
            best = max(row, key=lambda k: row[k])
            sw[f"tool{tid}_r_eff_{r_eff}"] = {
                "curve": row, "base_ceiling": round(cov0, 4), "argmax_dv": best,
                "argmax_ceiling": row[best], "gain": round(row[best] - cov0, 4),
                "missed_v_rows_at_dv0": [j for j in range(2 * nv) if not hit0[:, j].any()],
                "missed_v_rows_at_dv+0.02": [j for j in range(2 * nv) if not hit20[:, j].any()],
                "missed_v_rows_at_dv+0.03": [j for j in range(2 * nv) if not hit30[:, j].any()],
                "n_missed_cells_dv0": int((~hit0).sum()), "n_missed_cells_dv+0.02": int((~hit20).sum()),
            }
        out["sweep"][sname] = sw
    # does the +v leak alone (loop amplitude) or the r_eff dilation do the work? attribute:
    for sname, spec in SUITES.items():
        org, nu, nv = rig.scrub_grid(spec)
        base = rig.scrub_uv(spec, "trochoid")
        rows = [round(-(0.12 if spec["tank_shape"] == "elongated" else 0.18) / 2
                      + r * rig.CELL_M, 4) for r in range(max(1, int((0.12 if spec["tank_shape"] == "elongated" else 0.18) / rig.CELL_M)))]
        vspan = [min(v for _, v in base), max(v for _, v in base)]
        out.setdefault("attribution", {})[sname] = {
            "rows_v": rows, "plan_v_span_m": [round(vspan[0], 4), round(vspan[1], 4)],
            "patch_v_span_m": [round(org[1], 4), round(org[1] + 2 * nv * rig.FINE_M, 4)],
            "top_row_centre_v": round(org[1] + (2 * nv - 0.5) * rig.FINE_M, 4),
            "reach_with_r_eff_0.035": round(vspan[1] + 0.035, 4),
            "shortfall_m": round(org[1] + (2 * nv - 0.5) * rig.FINE_M - (vspan[1] + 0.035), 4),
        }
    p = os.path.join(REPO, "results", "aegis_v2", "I3_r242_mechanism.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
