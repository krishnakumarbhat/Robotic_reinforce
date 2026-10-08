"""I21 root-cause attribution, EXACT: reconstruct the historical v1 `fitted` verbatim
from commit ac8bc4b and attribute its coverage loss between its two defects.

v1 (verbatim, ac8bc4b lines 295-323):
    r_eff from the tool; p = 2*r_eff*0.8; half = 0.20; side = 0.12 | 0.18
    inset = r_eff                       <-- defect (2)
    rows: v = -side/2 + inset, += p, while v <= side/2 - inset   <-- defect (1): the
           walk stops at the FIRST row past the inset end, so the swept band is
           [v_start, v_start + p) regardless of how much patch is left
    u spans [-half+inset, half-inset]   <-- part of defect (2)
    C1 semicircle of radius p/2; resample 0.01

Variants, each changing ONE thing, scored with the rig's own fine-cell kernel:
    v1        verbatim
    v1_noU    inset removed from the u axis only  (row plan untouched)
    v1_noV    row plan spans the full patch, inset kept in u only
    v1_span   both fixed  == the shipped v2 rule
Inputs: none (no physics, no rig edit). Outputs: I21_r238_v1_attribution.json + table.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import kaggle_aegis_sweep as rig  # noqa: E402

SPECS = {"fixture_A": {"tank_shape": "round", "surface": "glossy",
                       "offset_cm": 0, "angle_deg": 0},
         "fixture_B": {"tank_shape": "elongated", "surface": "matte",
                       "offset_cm": 15, "angle_deg": 10}}
TOOLS = {0: 0.035, 1: 0.040, 2: 0.050}


def v1_polyline(spec, r_eff, inset_u=True, inset_v=True, span=False):
    """Purpose: the v1 polyline with one defect at a time disabled.
    inset_u=False -> u runs the full +-half (defect 2a removed)
    span=True     -> rows are re-placed at the band CENTRES of the inset sub-rect
                     (defect 1 removed: the walk spans the whole swept band)
    Inputs: spec, r_eff, defect switches. Outputs: (uv list, rows, pitch).
    """
    half = 0.20
    side = 0.12 if spec["tank_shape"] == "elongated" else 0.18
    inset = r_eff if inset_u else 0.0
    p = 2 * r_eff * 0.8
    v_start, v_end = -side * 0.5 + r_eff, side * 0.5 - r_eff
    if span:
        n = max(1, int(math.ceil((v_end - v_start) / p)))
        p = (v_end - v_start) / n
        rows = [v_start + (v_end - v_start) * (k + 0.5) / n for k in range(n)]
    else:
        rows, v = [], v_start
        while v <= v_end + 1e-6:
            rows.append(v)
            v += p
    uv = []
    n_line = max(2, int(2 * half / 0.01) + 1)
    for r, v in enumerate(rows):
        fwd = r % 2 == 0
        us = rig.np_linspace(-half + inset, half - inset, n_line)
        uv += [(float(u), v) for u in (us if fwd else us[::-1])]
        if r < len(rows) - 1:
            cx, cv, rad = (half - inset if fwd else -half + inset), v + p * 0.5, p * 0.5
            for k in range(1, 16):
                th = -math.pi / 2 + math.pi * k / 16
                uv.append((cx + (rad * math.cos(th) if fwd else -rad * math.cos(th)),
                           cv + rad * math.sin(th)))
    return rig._resample_uv(uv, 0.01), rows, p


def ceiling(spec, uv, r_eff):
    """Purpose: rig-kernel coverage_cont of a polyline dilated by r_eff + the rim vs
    interior split of the misses. Inputs: spec, uv, r_eff. Outputs: (cov, rim, interior).
    """
    import numpy as np
    org, nu, nv = rig.scrub_grid(spec)
    fu, fv = 2 * nu, 2 * nv
    cu = org[0] + (np.arange(fu) + 0.5) * rig.FINE_M
    cv = org[1] + (np.arange(fv) + 0.5) * rig.FINE_M
    G = np.stack(np.meshgrid(cu, cv, indexing="ij"), axis=-1).reshape(-1, 2)
    P = np.asarray(uv[::2] if len(uv) > 400 else uv, dtype=np.float64)
    hit = np.zeros(len(G), dtype=bool)
    for k in range(0, len(P), 256):
        hit |= (np.linalg.norm(G[:, None, :] - P[None, k:k + 256, :], axis=-1)
                <= r_eff).any(axis=1)
    m = (~hit).reshape(fu, fv)
    if not m.any():
        return float(hit.mean()), 0.0, 0.0
    edge = np.zeros_like(m)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return float(hit.mean()), float((m & edge).sum()) / m.sum(), float((m & ~edge).sum()) / m.sum()


def main():
    out = {"variants": {}}
    print(f"{'suite':10s} {'tool':4s} {'r_eff':6s} {'variant':9s} {'ceil':7s} {'rim/iss':8s} "
          f"{'rows':5s} {'pitch':7s} {'len_m':7s}")
    for sname, spec in SPECS.items():
        for tid, r_eff in TOOLS.items():
            for variant, kw in (("v1", {}), ("v1_noU", {"inset_u": False}),
                                ("v1_span", {"span": True}),
                                ("v1_fix", {"inset_u": False, "span": True}),
                                ("v2_ship", None)):
                if kw is None:      # the shipped rig path, scored by the same kernel
                    uv = rig.scrub_uv(dict(spec, r_eff=r_eff), "fitted")
                    _sp = 0.12 if spec["tank_shape"] == "elongated" else 0.18
                    _n = max(1, int(math.ceil(_sp / (2.0 * r_eff))))
                    rows = [0.0] * _n
                    p = _sp / _n
                else:
                    uv, rows, p = v1_polyline(spec, r_eff, **kw)
                cov, rim, iss = ceiling(spec, uv, r_eff)
                out["variants"][f"{sname}/tool{tid}/{variant}"] = {
                    "ceiling": round(cov, 4), "rim_frac_of_miss": round(rim, 4),
                    "interior_frac_of_miss": round(iss, 4), "n_rows": len(rows),
                    "pitch_m": round(p, 5), "path_len_m": round(rig.uv_length(uv), 4),
                    "r_eff": r_eff, "defects_disabled": sorted(kw or [])}
                print(f"{sname:10s} {tid:<4d} {r_eff:<6.3f} {variant:9s} {cov:<7.4f} "
                      f"{rim:.2f}/{iss:.2f}       {len(rows):<5d} {p:<7.4f} "
                      f"{rig.uv_length(uv):<7.4f}")
    print("\nmean ceiling over 3 tools, and the marginal recovery of each single fix:")
    for sname in SPECS:
        row = {}
        for variant in ("v1", "v1_noU", "v1_span", "v1_fix", "v2_ship"):
            c = [v["ceiling"] for k, v in out["variants"].items()
                 if k.startswith(sname) and k.endswith("/" + variant)]
            row[variant] = round(sum(c) / len(c), 4)
        base = row["v1"]
        print(f"  {sname}: v1 {base:.4f} | noU {row['v1_noU']:.4f} "
              f"(+{row['v1_noU'] - base:.4f}) | span {row['v1_span']:.4f} "
              f"(+{row['v1_span'] - base:.4f}) | both {row['v1_fix']:.4f} "
              f"(+{row['v1_fix'] - base:.4f}) | SHIPPED v2 {row['v2_ship']:.4f} "
              f"(+{row['v2_ship'] - base:.4f})")
        out.setdefault("per_suite", {})[sname] = row
    path = os.path.join(os.path.dirname(__file__), "I21_r238_v1_attribution.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
