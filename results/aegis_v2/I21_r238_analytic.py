"""I21 diagnosis: is `fitted` losing coverage to PITCH MATH, INSET, or the PHASE/servo?

Purpose: separate the three candidate root causes named in the I21 spec
    (a) pitch math leaving gaps, (b) inset rows missing edges, (c) phase bug
WITHOUT running physics, by scoring the *commanded polyline* through the rig's own
scoring kernel. A fine cell is covered iff some in-contact head position lies within
r_eff of its centre, so the polyline dilated by r_eff is the GEOMETRIC CEILING that no
amount of grip can exceed. Physics can only subtract from it.
Inputs: none (imports the frozen rig module; no rig edit, no physics).
Outputs: results/aegis_v2/I21_r238_analytic.json  +  a printed table.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import kaggle_aegis_sweep as rig  # noqa: E402

FIXTURES = {"fixture_A": {"tank_shape": "round", "surface": "glossy",
                           "offset_cm": 0, "angle_deg": 0},
            "fixture_B": {"tank_shape": "elongated", "surface": "matte",
                          "offset_cm": 15, "angle_deg": 10}}
TOOLS = {0: ("box", [0.05, 0.035, 0.012]), 1: ("box", [0.04, 0.04, 0.030]),
         2: ("box", [0.09, 0.05, 0.006])}


def ceiling(spec, mode, r_eff):
    """Purpose: coverage_cont of the commanded polyline = geometric ceiling.
    Uses the rig's own FINE_M grid and the same `d <= r_eff` test as _coverage_cont.
    Inputs: fixture spec, path mode, footprint radius. Outputs: (coverage, cell map).
    """
    import numpy as np
    org, nu, nv = rig.scrub_grid(spec)
    fu, fv = 2 * nu, 2 * nv
    cu = org[0] + (np.arange(fu) + 0.5) * rig.FINE_M
    cv = org[1] + (np.arange(fv) + 0.5) * rig.FINE_M
    G = np.stack(np.meshgrid(cu, cv, indexing="ij"), axis=-1).reshape(-1, 2)
    uv = rig.scrub_uv(dict(spec, r_eff=r_eff), mode)
    P = np.asarray(uv[::2] if len(uv) > 400 else uv, dtype=np.float64)
    hit = np.zeros(len(G), dtype=bool)
    for k in range(0, len(P), 256):
        d = np.linalg.norm(G[:, None, :] - P[None, k:k + 256, :], axis=-1)
        hit |= (d <= r_eff).any(axis=1)
    return float(hit.mean()), hit.reshape(fu, fv)


def miss_profile(hit, spec):
    import numpy as np
    """Purpose: WHERE are the missed fine cells -- interior or rim?  A pitch bug shows
    up as interior stripes (uncovered cells BETWEEN covered rows); an inset/edge bug
    shows up as a fully uncovered outer band.  Reports the fraction of missed cells in
    the outer ring vs the interior, plus per-column (u) and per-row (v) miss counts.
    Inputs: hit map (fu, fv), spec. Outputs: dict.
    """
    fu, fv = hit.shape
    miss = ~hit
    if not miss.any():
        return {"miss_frac": 0.0, "rim_frac_of_miss": 0.0, "interior_frac_of_miss": 0.0,
                "per_v_miss": [0] * fv, "per_u_miss": [0] * fu}
    edge = np.zeros_like(miss)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    rim = miss & edge
    interior = miss & ~edge
    return {"miss_frac": round(float(miss.mean()), 4),
            "rim_frac_of_miss": round(float(rim.sum()) / miss.sum(), 4),
            "interior_frac_of_miss": round(float(interior.sum()) / miss.sum(), 4),
            "per_v_miss": [int(x) for x in miss.sum(axis=0)],
            "per_u_miss": [int(x) for x in miss.sum(axis=1)]}


def main():
    out = {"rig_version": 2, "cell_m": rig.CELL_M, "fine_m": rig.FINE_M,
           "note": "analytic geometric ceiling of the COMMANDED polyline, scored with the "
                   "rig's own fine-cell kernel; physics can only subtract from it",
           "modes": {}}
    print(f"{'suite':10s} {'tool':4s} {'r_eff':6s} {'mode':9s} {'ceil':7s} "
          f"{'len_m':7s} {'rows':5s} {'pitch':7s} {'rim/iss':8s}")
    for sname, spec in FIXTURES.items():
        for tid, (_k, half) in TOOLS.items():
            r_eff = float(min(half[0], half[1]))
            side = 0.12 if spec["tank_shape"] == "elongated" else 0.18
            for mode in ("raster", "trochoid", "fitted"):
                cov, hit = ceiling(spec, mode, r_eff)
                uv = rig.scrub_uv(dict(spec, r_eff=r_eff), mode)
                L = rig.uv_length(uv)
                prof = miss_profile(hit, spec)
                if mode == "fitted":
                    n_rows = max(1, int(math.ceil(side / (2.0 * r_eff))))
                    pitch = round(side / n_rows, 5)
                else:
                    n_rows, pitch = 0, 0.0
                key = f"{sname}/tool{tid}/{mode}"
                out["modes"][key] = {"ceiling": round(cov, 4), "path_len_m": round(L, 4),
                                     "n_rows": n_rows, "pitch_m": pitch,
                                     "r_eff": r_eff, **prof}
                print(f"{sname:10s} {tid:<4d} {r_eff:<6.3f} {mode:9s} {cov:<7.4f} "
                      f"{L:<7.4f} {n_rows:<5d} {pitch:<7.4f} "
                      f"{prof['rim_frac_of_miss']:.2f}/{prof['interior_frac_of_miss']:.2f}")
    # headline: does fitted beat trochoid on the CEILING (geometry) at all?
    d = {}
    for sname, spec in FIXTURES.items():
        ce_f = [v["ceiling"] for k, v in out["modes"].items()
                if k.startswith(sname) and k.endswith("/fitted")]
        ce_t = [v["ceiling"] for k, v in out["modes"].items()
                if k.startswith(sname) and k.endswith("/trochoid")]
        lf = [v["path_len_m"] for k, v in out["modes"].items()
              if k.startswith(sname) and k.endswith("/fitted")]
        lt = [v["path_len_m"] for k, v in out["modes"].items()
              if k.startswith(sname) and k.endswith("/trochoid")]
        d[sname] = {"fitted_ceiling": ce_f, "trochoid_ceiling": ce_t,
                    "fitted_len": lf, "trochoid_len": lt,
                    "ceiling_delta": round(sum(ce_f) / 3 - sum(ce_t) / 3, 4),
                    "len_ratio": round((sum(lf) / 3) / (sum(lt) / 3), 4)}
    out["per_suite"] = d
    print("\nceiling deltas (fitted - trochoid, mean over 3 tools):")
    for s, v in d.items():
        print(f"  {s}: dceil {v['ceiling_delta']:+.4f}  len_ratio {v['len_ratio']:.4f}")
    path = os.path.join(os.path.dirname(__file__), "I21_r238_analytic.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
