"""N481 phase 0b — RAY-EXACT replay of the depth-registration cast (no coverage arithmetic).

Purpose: confirm the N481.1 attribution of the elongated-face `reg_err_xy` residual with the
REAL ray caster, so the prediction is exact rather than a grid approximation. It rebuilds the
same world `PyBulletScrub.__init__` builds (floor plane, fixture body at `fixture_pose(spec)`
in the TRUE frame, tool box parked axis-aligned at `[0, 0, 0.5]`), casts the same n x n grid
with the same filters `_cast` applies, and prints the resulting `reg_err_xy` for any
(pad_hu, pad_hv, H, n) dose. Nothing here computes coverage_cont or success (G7): it is a
replay of the ESTIMATOR only, and every claim it supports is checked against the rig's own
logged `reg_err_xy_m` / `reg_cast_pts` in results/aegis_v2/.

Inputs: nothing (self-check against an archived file). Outputs: table on stdout.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pybullet as p
import pybullet_data

FACE_HU, FACE_HV = 0.34, 0.14                  # frozen elongated halfExtents
TOOL_Z = 0.5                                   # frozen tool basePosition z
TOOL_TH = (0.012, 0.030, 0.006)                # frozen per-tool half thickness (z only)


def cast(spec: dict, noise: tuple, half: float, n: int, off: tuple,
         pad: tuple, cid_client: int = 0) -> dict:
    """One `_cast` replay. Returns {npts, mean_x, mean_y, norm, points}.

    Inputs: fixture spec, (dx, dy, dyaw) plan noise, window half H, ray count n, lattice
    offset (ox, oy), tool pad half-extents (hu, hv). Outputs: the kept top-face point set and
    the sample-mean residual against the true face centre.
    """
    fx = round(float(spec.get("offset_cm", 0)) / 100.0, 4)
    fy = round(float(spec.get("offset_y_cm", 0)) / 100.0, 4)
    yaw = math.radians(float(spec.get("angle_deg", 0)))
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid_client)
    plane = p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE),
                              basePosition=[0, 0, 0], physicsClientId=cid_client)
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    shape = (p.createCollisionShape(p.GEOM_BOX, halfExtents=[FACE_HU, FACE_HV, 0.12])
             if spec.get("tank_shape") == "elongated"
             else p.createCollisionShape(p.GEOM_CYLINDER, radius=FACE_HU, height=0.44))
    fixture = p.createMultiBody(0, shape, basePosition=[fx, fy, 0.15], baseOrientation=quat,
                                physicsClientId=cid_client)
    tool = p.createMultiBody(0.1, p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[pad[0], pad[1], pad[2]]),
        basePosition=[0, 0, TOOL_Z], physicsClientId=cid_client)
    cx, cy = fx + noise[0], fy + noise[1]
    z_hi, z_lo = 0.15 + 0.8, 0.15 - 0.05
    xs = np.linspace(cx - half + off[0], cx + half + off[0], int(n))
    ys = np.linspace(cy - half + off[1], cy + half + off[1], int(n))
    X, Y = np.meshgrid(xs, ys)
    frm = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_hi)], axis=-1).tolist()
    to = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_lo)], axis=-1).tolist()
    res = []
    for i in range(0, len(frm), 1024):
        res.extend(p.rayTestBatch(frm[i:i + 1024], to[i:i + 1024], physicsClientId=cid_client))
    p.disconnect(physicsClientId=cid_client)
    hits = [(r[3][0], r[3][1], r[3][2]) for r in res
            if r[0] not in (-1, tool, plane)]
    if not hits:
        return {"npts": 0, "norm": float("nan")}
    zmax = max(h[2] for h in hits)
    pts = [(h[0], h[1]) for h in hits if abs(h[2] - zmax) < 0.01]
    mx = sum(q[0] for q in pts) / len(pts) - fx
    my = sum(q[1] for q in pts) / len(pts) - fy
    return {"npts": len(pts), "mean_x": mx, "mean_y": my, "norm": math.hypot(mx, my)}


def selfcheck(path: str) -> dict:
    """Replay the archived candidate arm and compare to the rig's own reg_err_xy / reg_cast_pts."""
    rows = [json.loads(line) for line in open(path)]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    has_cmp = any(r.get("record") == "compare" for r in rows)
    per = len(eps) // (2 if has_cmp else 1)
    half, n = float(hdr["aegis_reg_half_m"]), int(float(hdr["aegis_reg_n"]))
    pads = ((0.05, 0.035, TOOL_TH[0]), (0.04, 0.04, TOOL_TH[1]), (0.09, 0.05, TOOL_TH[2]))
    worst_norm = worst_pts = 0.0
    checked = 0
    for r in eps[:per]:
        if r.get("suite") != "fixture_B" or not r.get("reg_ok"):
            continue
        out = cast(r["fixture_spec"], tuple(r["pose_noise"]), half, n,
                   tuple(r["reg_cast_win_m"]), pads[int(r["tool_id"]) % 3])
        checked += 1
        worst_pts = max(worst_pts, abs(out["npts"] - r["reg_cast_pts"]))
        d = abs(out["norm"] - float(r["reg_err_xy_m"]))
        if d == d:                      # NaN-safe
            worst_norm = max(worst_norm, d)
    return {"file": path.split("/")[-1], "H": half, "n": n, "episodes": checked,
            "max_abs_npts_err": worst_pts, "max_abs_norm_err_mm": round(1e3 * worst_norm, 9)}


def ladder(spec: dict, half: float, off: tuple, pads: dict, n_list=(32, 64, 128)) -> dict:
    """Predicted residual (mm) at zero plan offset for each pad geometry and ray count."""
    out = {}
    for name, pad in pads.items():
        out[name] = {str(n): round(1e3 * cast(spec, (0.0, 0.0, 0.0), half, n, off, pad)["norm"], 4)
                     for n in n_list}
    return out


def main() -> int:
    spec_b = {"tank_shape": "elongated", "surface": "matte", "offset_cm": 15, "angle_deg": 10}
    frozen = {"sponge(0,1)": (0.05, 0.035, 0.012), "brush(2)": (0.04, 0.04, 0.030),
              "mop(3)": (0.09, 0.05, 0.006)}
    doses = dict(frozen, **{"all-mop": (0.09, 0.05, 0.006), "floor(0.05,0.0275)": (0.05, 0.0275, 0.012),
                            "half-hu(0.025,0.05)": (0.025, 0.05, 0.012)})
    report = {"selfcheck": [selfcheck(f) for f in sys.argv[1:]] or "none",
              "prediction_zero_offset_mm": ladder(spec_b, 0.6, (0.0, 0.0), doses)}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())