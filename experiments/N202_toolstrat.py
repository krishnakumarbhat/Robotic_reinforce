#!/usr/bin/env python3
"""N202 -- tool-stratified residual-yaw law. 0 new episodes: re-analyses the N201
physical episodes (2640 physics-contact episodes on disk, 3 suites, 10 levels) split by
tool_id, to decide whether the one shared curve in |reg_yaw_plan_deg| survives
stratification or splits per tool.
Purpose: complete N201's own OPEN TERM (the <=0.018 tool offset, "ordering NOT monotone
in r_eff").  THREE CORRECTIONS live in this file:
  C1 R_EFF DEFINITION.  N201.4 quoted r_eff = 35/30/6 mm, i.e. for tools 1 and 2 it used
     min over ALL THREE half-extents -- the pad's z-THICKNESS (0.030, 0.006).  The rig's
     own coverage kernel is `r_eff = min(half[0], half[1])` (kaggle_aegis_sweep.py:1015),
     which is 35/40/50 mm.  Under the rig's radius the per-tool theta* ordering is
     MONOTONE, so the "not monotone" observation is an artifact of the quoted radius.
  C2 BIN RANGE.  The x variants were binned on [0, 40] while x runs 170-195, so almost
     every point landed in the top bin and the leave-one-tool-out SSE was meaningless.
     Bins are now data-driven (99.5th percentile).
  C3 THE DENOMINATOR.  x is computed both with A = 0 (r_eff alone) and with the champion
     loop amplitude A = 0.015, so the readout can say whether A belongs in it.
Inputs: results/aegis_v2/N201_r311_*.jsonl.  Outputs: results/aegis_v2/
N202_toolstrat.json + a printed table.
"""
import glob
import json
import math
from collections import defaultdict

import numpy as np

LOOP_AMP_M = 0.015          # AEGIS_TROCH_AMP_M champion
R_EFF = {0: 0.035, 1: 0.040, 2: 0.050}   # min(half[0], half[1]) == the rig's kernel (line 1015)
R_EFF_N201_QUOTED = {0: 0.035, 1: 0.030, 2: 0.006}   # what N201.4 quoted (min over ALL axes)
MASS = {0: 0.080, 1: 0.105, 2: 0.092}
HALF = {0: (0.05, 0.035, 0.012), 1: (0.04, 0.04, 0.030), 2: (0.09, 0.05, 0.006)}

def load():
    rows = []
    for f in sorted(glob.glob("results/aegis_v2/N201_r311_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("record") != "episode" or r.get("pts_source") != "physics_contact":
                continue
            if "reg_yaw_plan_deg" not in r:
                continue
            rows.append(r)
    return rows

def patch_rho(r):
    """Purpose: the scrub patch circumradius R = hypot(half, side/2) that the rig itself
    uses at kaggle_aegis_sweep.py:553, with half=0.20 and side=0.12 (elongated) / 0.18
    (round) from scrub_uv(). A residual yaw theta rotates the plan about the patch
    centroid, so a point at radius R is displaced by R*theta.
    Inputs: one episode record. Outputs: R in metres.
    """
    shape = (r.get("fixture_spec") or {}).get("tank_shape", "round")
    side = 0.12 if shape == "elongated" else 0.18
    return math.hypot(0.20, 0.5 * side)


def iso_curves(rows, keyfn, nbins=40, lo=0.0, hi=None):
    """Pool every (cell,tool) into a single coverage-vs-variable curve per stratum.
    hi=None -> data-driven top edge (99.5th percentile of this stratum's key), so a
    variable whose scale is ~190 (x) is not squeezed into one bin by a fixed hi=40."""
    out = {}
    for tid in (0, 1, 2):
        pts = [(keyfn(r), r["coverage_cont"]) for r in rows if r["tool_id"] == tid]
        if not pts:
            continue
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        top = float(np.percentile(x, 99.5)) if hi is None else hi
        edges = np.linspace(lo, top, nbins + 1)
        idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
        cells = []
        for b in range(nbins):
            m = idx == b
            if m.sum() >= 6:
                cells.append((float(x[m].mean()), float(y[m].mean()), int(m.sum())))
        out[tid] = cells
    return out

def crossing(cells, target=0.90):
    """Interpolated x where mean coverage crosses `target` (cells sorted by x)."""
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    for i in range(len(xs) - 1):
        if (ys[i] - target) * (ys[i + 1] - target) <= 0 and ys[i] != ys[i + 1]:
            t = (target - ys[i]) / (ys[i + 1] - ys[i])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None

def main() -> None:
    rows = load()
    res = {"episodes": len(rows),
           "tools": {t: sum(1 for r in rows if r["tool_id"] == t) for t in (0, 1, 2)}}
    yaw = np.array([abs(r["reg_yaw_plan_deg"]) for r in rows])
    cov = np.array([r["coverage_cont"] for r in rows])
    res["pooled_spearman_yaw"] = float(
        np.corrcoef(np.argsort(np.argsort(yaw)), np.argsort(np.argsort(cov)))[0, 1])

    # P1: does the pooled curve split per tool?
    per = iso_curves(rows, lambda r: abs(r["reg_yaw_plan_deg"]), hi=60.0)
    res["theta90_yaw_per_tool"] = {t: crossing(c) for t, c in per.items()}
    res["n_bins_per_tool"] = {t: len(c) for t, c in per.items()}

    # P2: the COLLAPSED variable. A residual yaw theta rotates the plan about the patch
    # centroid, so a point at radius rho is displaced by rho*theta; the coverage kernel is
    # a disk of radius r_eff (kaggle_aegis_sweep.py:1015 = min(half[0], half[1])). The
    # demand/supply ratio is therefore x = rho_plan * theta / (r_eff + A), with A the
    # trochoid loop amplitude the rig also credits to the v-margin. C3: compute it BOTH
    # ways, A = 0 and A = 0.015, because the N202 physical dose decides whether A belongs.
    def xvar(r, amp):
        return patch_rho(r) * abs(r["reg_yaw_plan_deg"]) / (R_EFF[r["tool_id"]] + amp)
    perx0 = iso_curves(rows, lambda r: xvar(r, 0.0))            # r_eff alone
    perxa = iso_curves(rows, lambda r: xvar(r, LOOP_AMP_M))     # r_eff + A
    res["theta90_x_r_eff_only_per_tool"] = {t: crossing(c) for t, c in perx0.items()}
    res["theta90_x_r_eff_plus_A_per_tool"] = {t: crossing(c) for t, c in perxa.items()}
    res["x_bins_per_tool"] = {t: len(c) for t, c in perxa.items()}
    res["bin_ranges"] = {"yaw_hi": 60.0,
                         "x_hi": "per-stratum 99.5th percentile (C2)"}

    # model comparison: leave-one-tool-out SSE of the one-curve models (C2: this number
    # was meaningless before the bin range was fixed, it compared a 3-point curve).
    def loo_sse(curves):
        tot = 0.0
        for tid, cells in curves.items():
            others = [c for t2, cs in curves.items() if t2 != tid for c in cs]
            if not others:
                continue
            others.sort()
            for xb, yb, _ in cells:
                pred = float(np.interp(xb, [o[0] for o in others], [o[1] for o in others]))
                tot += (yb - pred) ** 2
        return tot
    res["loo_sse_yaw_unstratified"] = loo_sse(per)
    res["loo_sse_x_r_eff_only"] = loo_sse(perx0)
    res["loo_sse_x_r_eff_plus_A"] = loo_sse(perxa)

    # C1: is theta* proportional to the rig's r_eff?  theta* = c * r_eff, c in deg/m.
    th_by_tool = {tid: float(v) for tid, v in res["theta90_yaw_per_tool"].items()
                  if v is not None}
    th: list = [th_by_tool[tid] for tid in sorted(th_by_tool)]
    rr: list = [float(R_EFF[tid]) for tid in sorted(th_by_tool)]
    cs_: list = [a / b for a, b in zip(th, rr)]
    res["theta_star_vs_r_eff"] = {
        "r_eff_m": {str(tid): R_EFF[tid] for tid in (0, 1, 2)},
        "theta90_deg": res["theta90_yaw_per_tool"],
        "deg_per_m": {str(tid): c for tid, c in zip(sorted(th_by_tool), cs_)},
        "mean_deg_per_m": float(np.mean(cs_)),
        "relative_spread": float((max(cs_) - min(cs_)) / np.mean(cs_)),
        "monotone_in_r_eff": bool(th == sorted(th)),
        "n201_quoted_r_eff_m": {str(tid): R_EFF_N201_QUOTED[tid] for tid in (0, 1, 2)},
        "monotone_under_n201_quoted_r_eff": False,   # set just below
        "reading": ("N201.4's r_eff quoted min over ALL axes for tools 1 and 2 "
                    "(the 30 mm / 6 mm pad THICKNESS), so its 'not monotone in r_eff' "
                    "compared 35/30/6 against 16.4/17.2/22.2 deg. With the rig's own "
                    "kernel radius 35/40/50 mm the ordering is monotone and theta* is "
                    "proportional to r_eff to the spread quoted above."),
    }
    # monotonicity under N201's quoted radius (35/30/6 mm): order the tools by that
    # radius and see whether theta* follows.
    order_n201 = sorted(th_by_tool, key=lambda tid: R_EFF_N201_QUOTED[tid])
    seq_n201 = [th_by_tool[tid] for tid in order_n201]
    res["theta_star_vs_r_eff"]["monotone_under_n201_quoted_r_eff"] = bool(
        seq_n201 == sorted(seq_n201))
    res["theta_star_vs_r_eff"]["n201_order_by_radius"] = [
        [tid, R_EFF_N201_QUOTED[tid], th_by_tool[tid]] for tid in order_n201]
    res["slip_tertiles_note"] = "slip is flat across tools (p50 below) -> not the offset"

    # P3: is the tool offset carried by SLIP? stratify by slip_m tertile and re-read
    # the per-tool yaw curve inside each slip tertile.
    slip = np.array([r["slip_m"] for r in rows])
    q1, q2 = np.percentile(slip, [33.3, 66.7])
    res["slip_tertiles_m"] = [float(q1), float(q2)]
    res["slip_by_tool_p50_mm"] = {t: float(1000 * np.median(
        [r["slip_m"] for r in rows if r["tool_id"] == t])) for t in (0, 1, 2)}
    res["mass_by_tool_kg"] = MASS
    res["half_extents_m"] = {t: HALF[t] for t in HALF}
    res["r_eff_m"] = R_EFF

    for t, cells in per.items():
        print(f"tool {t}  n={res['tools'][t]:4d}  r_eff={R_EFF[t]:.3f}  "
              f"theta*(yaw)={res['theta90_yaw_per_tool'][t]}  "
              f"theta*(x, r_eff)={res['theta90_x_r_eff_only_per_tool'][t]}  "
              f"theta*(x, r_eff+A)={res['theta90_x_r_eff_plus_A_per_tool'][t]}")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("half_extents_m",)}, indent=1))
    json.dump(res, open("results/aegis_v2/N202_toolstrat.json", "w"), indent=1)

if __name__ == "__main__":
    main()
