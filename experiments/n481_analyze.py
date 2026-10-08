"""N481 phase 0 — EXACT offline reconstruction of the depth-registration estimator.

Purpose: attribute the unattributed `reg_err_xy` residual on the elongated metric face
(N478d) by rebuilding the estimator from archived fields alone -- no physics, no PyBullet,
and NO arithmetic on coverage_cont/success (G7: only the rig produces those).

Inputs: archived rig JSONL under results/aegis_v2/ whose header carries aegis_reg == "depth".
Outputs: per-file table of predicted vs measured estimator residual (the exactness check is
the reconstructed top-point COUNT, which the rig also logs) plus the n-ladder that decides
the pre-registered discriminator D1b.

The reconstruction is EXACT on the flat top face: every ray in `_cast` is vertical, so a hit's
(x, y) IS its grid coordinate; the only non-tool/non-plane body is the fixture itself, so the
1 cm max-z band keeps every hit and the rim/pedestal cell D3 cannot contribute; and the mean
estimator is a plain sample mean over the grid points inside the rotated rectangle.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

RES = "results/aegis_v2"
FACE_HU, FACE_HV = 0.34, 0.14      # frozen elongated box halfExtents (FACE_HU_M = FACE_HV_M = 0)
# Frozen tool pads (PyBulletScrub.TOOL_SHAPES, tool_id = seed % 3). N481.1: the head is
# created at the WORLD ORIGIN, axis-aligned (`createMultiBody(..., basePosition=[0, 0, 0.5])`,
# no quaternion) and it is STILL THERE when the registration cast runs (the cast happens
# before `rig.run()`), so it shadows an axis-aligned patch of the ray grid and `_cast` drops
# every ray it intercepts (`r[0] != rig.tool`). fixture_B's body sits at x = +0.15 m, so the
# shadow lands OFF the face centre; fixture_A's body is AT the origin, where a centred patch
# cancels. That is the elongated-only residual.
TOOL_PAD = ((0.05, 0.035), (0.04, 0.04), (0.09, 0.05))


def fixture_xy(spec: dict) -> tuple[float, float]:
    """World xy of the fixture body centre, i.e. of the top-face centre, from frozen spec fields."""
    return (float(spec.get("offset_cm", 0)) / 100.0, float(spec.get("offset_y_cm", 0)) / 100.0)


def reconstruct(rec: dict, half: float, n: int, censor: bool = True) -> dict:
    """Rebuild the top-point set, the frozen `mean` estimator and N190's `extent` estimator.

    Inputs: episode record (fixture_spec, pose_noise, tool_id, reg_cast_win_m), H, n, censor flag.
    Outputs: dict(npts, npts_uncensored, mean_xy, mean_norm, extent_xy, extent_norm, margin).

    Geometry, read off the rig source (no assumption left):
      * the fixture BODY keeps the true frame -- only the PLAN is perturbed (`build_plan` adds
        noise to the path origin and yaw), so the top face sits at `fixture_xy(spec)` with yaw
        `angle_deg` and the pose noise NEVER rotates the face;
      * the head is parked at the WORLD ORIGIN with no yaw, and the cast runs before
        `rig.run()`, so `_cast` drops every ray the head shadows: an axis-aligned pad-footprint
        patch centred on (0, 0), independent of the plan.
    """
    spec, noise = rec["fixture_spec"], rec["pose_noise"]
    fx, fy = fixture_xy(spec)
    yaw = math.radians(float(spec.get("angle_deg", 0.0)))          # face yaw: TRUE, unperturbed
    cx, cy = fx + float(noise[0]), fy + float(noise[1])            # plan origin == grid centre
    ox, oy = (float(v) for v in rec["reg_cast_win_m"])
    xs = np.linspace(cx - half + ox, cx + half + ox, int(n))
    ys = np.linspace(cy - half + oy, cy + half + oy, int(n))
    X, Y = np.meshgrid(xs, ys)
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx, dy = X.ravel() - fx, Y.ravel() - fy
    u, v = dx * c - dy * s, dx * s + dy * c
    inside = (np.abs(u) <= FACE_HU) & (np.abs(v) <= FACE_HV)
    n_face = int(inside.sum())
    if censor:
        th, tv = TOOL_PAD[int(rec.get("tool_id", 0)) % 3]
        inside = inside & ~((np.abs(X.ravel()) <= th) & (np.abs(Y.ravel()) <= tv))
    px, py = X.ravel()[inside], Y.ravel()[inside]
    if px.size == 0:
        nan = float("nan")
        return {"npts": 0, "n_face": n_face, "mean": (nan, nan, nan), "extent": (nan, nan, nan)}
    mx, my = float(px.mean() - fx), float(py.mean() - fy)
    qx, qy = px * c - py * s, px * s + py * c
    ex, ey = 0.5 * (qx.max() + qx.min()), 0.5 * (qy.max() + qy.min())
    ex, ey = ex * c - ey * s, ex * s + ey * c           # rotate back to world (R(-yaw)^T = R(yaw))
    # window border margin of the kept points: >= a_slack means the window CONTAINS the face
    margin = min(float(px.min() - (cx - half + ox)), float((cx + half + ox) - px.max()),
                 float(py.min() - (cy - half + oy)), float((cy + half + oy) - py.max()))
    return {"npts": int(px.size), "n_face": n_face, "mean": (mx, my, float(math.hypot(mx, my))),
            "extent": (ex, ey, float(math.hypot(ex, ey))), "margin": margin}


def load(path: str) -> tuple[dict, list[dict], dict]:
    rows = [json.loads(line) for line in open(path)]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    return hdr, eps, next((r for r in rows if r.get("record") == "compare"), {})


def arm_split(eps: list[dict], n_arms: int) -> list[list[dict]]:
    """The rig emits one contiguous block per arm (candidate block, then baseline block)."""
    per = len(eps) // n_arms
    return [eps[i * per:(i + 1) * per] for i in range(n_arms)]


def pct(vals: list[float], q: float) -> float:
    return float(np.percentile(vals, q)) if vals else float("nan")


def study(path: str, verbose: bool = False) -> dict:
    """Reconstruct the candidate arm's fixture_B episodes and compare with the rig's own log."""
    hdr, eps, cmp_ = load(path)
    arms = arm_split(eps, 2 if cmp_ else 1)
    cand = [r for r in arms[0] if r.get("suite") == "fixture_B" and r.get("reg_ok")
            and r.get("reg_err_xy_m") is not None]
    half = float(hdr["aegis_reg_half_m"])
    n = int(float(hdr["aegis_reg_n"]))
    meas, pred, ext, rows = [], [], [], []
    exact_npts = exact_norm = 0
    for r in cand:
        out = reconstruct(r, half, n)
        mm = float(r["reg_err_xy_m"])
        meas.append(mm)
        if out["npts"] == r.get("reg_cast_pts"):
            exact_npts += 1
        if out["npts"] and abs(out["mean"][2] - mm) < 1e-9:
            exact_norm += 1
        if out["npts"]:
            pred.append(out["mean"][2])
            ext.append(out["extent"][2])
            rows.append({"seed": r["seed"], "meas_mm": round(1e3 * mm, 4),
                         "pred_mm": round(1e3 * out["mean"][2], 4),
                         "pred_x_mm": round(1e3 * out["mean"][0], 4),
                         "pred_y_mm": round(1e3 * out["mean"][1], 4),
                         "ext_mm": round(1e3 * out["extent"][2], 4),
                         "npts": out["npts"], "npts_rig": r.get("reg_cast_pts"),
                         "win_margin_mm": round(1e3 * out.get("margin", float("nan")), 2),
                         "cn_margin_mm": (None if r.get("reg_cn_margin") is None
                                          else round(1e3 * float(r["reg_cn_margin"]), 2))})
    res = {"file": os.path.basename(path), "pose_noise_cfg": hdr.get("pose_noise_cfg"),
           "n": n, "H": half, "pitch_mm": round(2e3 * half / (n - 1), 3),
           "episodes": len(cand), "npts_exact": exact_npts, "norm_exact": exact_norm,
           "meas_p50_mm": round(1e3 * pct(meas, 50), 3), "meas_p90_mm": round(1e3 * pct(meas, 90), 3),
           "meas_max_mm": round(1e3 * max(meas), 3) if meas else None,
           "pred_p50_mm": round(1e3 * pct(pred, 50), 3), "pred_p90_mm": round(1e3 * pct(pred, 90), 3),
           "pred_max_mm": round(1e3 * max(pred), 3) if pred else None,
           "ext_pred_p50_mm": round(1e3 * pct(ext, 50), 3),
           "ext_pred_p90_mm": round(1e3 * pct(ext, 90), 3),
           "ext_pred_max_mm": round(1e3 * max(ext), 3) if ext else None,
           "pred_over_halfpitch_max": round(max(pred) / (half / (n - 1)), 4) if pred else None}
    if verbose:
        res["rows"] = rows
    return res


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv[1:]
    files = args or sorted(os.path.join(RES, f) for f in os.listdir(RES)
                           if f.startswith("N478c_r496_n") and f.endswith(".jsonl"))
    print(json.dumps([study(f, verbose) for f in files], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())