"""Purpose: N191 diagnostic -- (a) how often the frozen PCA yaw test declares the elongated
top face UNOBSERVABLE and silently falls back to the raw noisy prior, and (b) whether a
truncation-robust minimum-area-rectangle (minrect) yaw estimator is unbiased where PCA is not.
Re-creates the rig's EXACT grid, top-face test and PCA branch; diagnostic only (no keep/discard
may cite it -- the decider must come from experiments/kaggle_aegis_sweep.py).
Inputs: none. Outputs: results/aegis_v2/N191_probe.json + stdout.
"""
import json
import math
import os
import random
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

import kaggle_aegis_sweep as K  # noqa: E402

SUITES = ("fixture_B",)
SIGMAS = (("0", "0"), ("0.16", "32"), ("0.20", "40"), ("0.24", "48"), ("0.28", "56"))
N_SEEDS = 40
HALF = 0.6
REG_N = 32


def top_points(cx, cy, z_hi, z_lo, n, half, rig):
    """Purpose: the rig's exact top-face point set for an n x n grid of half-extent `half`.
    Inputs: grid centre, z bracket, resolution, half extent, rig. Outputs: Nx2 (x, y) array.
    """
    xs = np.linspace(cx - half, cx + half, n)
    ys = np.linspace(cy - half, cy + half, n)
    X, Y = np.meshgrid(xs, ys)
    frm = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_hi)], axis=-1).tolist()
    to = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_lo)], axis=-1).tolist()
    res = []
    for i in range(0, len(frm), 1024):  # pybullet hard-caps one batch at 1024 rays
        res.extend(rig.p.rayTestBatch(frm[i:i + 1024], to[i:i + 1024], physicsClientId=0))
    hits = [(r[3][0], r[3][1], r[3][2]) for r in res
            if r[0] != -1 and r[0] != rig.tool and r[0] != rig.plane]
    if not hits:
        return np.zeros((0, 2))
    zm = max(h[2] for h in hits)
    return np.array([(h[0], h[1]) for h in hits if abs(h[2] - zm) < 0.01])


def pca_yaw(pts, prior):
    """Purpose: the rig's FROZEN yaw branch, verbatim, including the degeneracy test.
    Inputs: Nx2 points, prior yaw (rad). Outputs: (yaw_est or None, degenerate flag).
    """
    if pts.shape[0] <= 2:
        return None, True
    cov = np.cov(pts[:, 0], pts[:, 1])
    vals, vecs = np.linalg.eigh(cov)
    if vals[1] < 1e-9 or vals[0] / max(vals[1], 1e-12) > 0.8:
        return None, True
    axis = vecs[:, 1] if vals[1] >= vals[0] else vecs[:, 0]
    yaw = math.atan2(axis[1], axis[0])
    while yaw - prior > math.pi / 2:
        yaw -= math.pi
    while yaw - prior < -math.pi / 2:
        yaw += math.pi
    return yaw, False


def minrect_yaw(pts, prior, span_pitch):
    """Purpose: truncation-robust yaw = orientation minimising the 2-D bounding-box area.
    A rectangle's minimum-area enclosing orientation is its edge orientation, and deleting
    points (ray-grid truncation) cannot move the minimiser: every off-edge rotation strictly
    inflates one of the two extents. Needs no prior except the same pi-branch snap.
    Inputs: Nx2 points, prior yaw (rad), grid pitch (m). Outputs: yaw or None if degenerate.
    """
    if pts.shape[0] < 4:
        return None
    ext = pts.max(axis=0) - pts.min(axis=0)
    if min(ext) < 2.0 * span_pitch:   # scale-free degeneracy: the set is thinner than the grid
        return None
    phis = np.linspace(0.0, math.pi, 180, endpoint=False)
    c, s = np.cos(-phis), np.sin(-phis)
    qx = pts[:, 0][None, :] * c[:, None] - pts[:, 1][None, :] * s[:, None]
    qy = pts[:, 0][None, :] * s[:, None] + pts[:, 1][None, :] * c[:, None]
    area = (qx.max(axis=1) - qx.min(axis=1)) * (qy.max(axis=1) - qy.min(axis=1))
    yaw = float(phis[int(np.argmin(area))])
    while yaw - prior > math.pi / 2:
        yaw -= math.pi
    while yaw - prior < -math.pi / 2:
        yaw += math.pi
    return yaw


def wrap(a):
    """Purpose: wrap an angle difference into (-pi, pi]. Inputs: rad. Outputs: rad.
    """
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def main():
    """Purpose: run the (a)/(b) diagnostic across the sigma ladder on fixture_B.
    Inputs: none. Outputs: results/aegis_v2/N191_probe.json.
    """
    out = {}
    for sig, sigdeg in SIGMAS:
        K.POSE_NOISE = f"{sig},{sigdeg}"
        rows = []
        for suite in SUITES:
            for seed in range(N_SEEDS):
                spec = dict(K.FIXTURES[suite])
                noise = K.pose_noise(random.Random(seed * 104729 + 3))
                friction = K.sample_friction(random.Random(seed), spec["surface"])
                rig = K.PyBulletScrub(spec, friction, seed % 3, K.T_MAX, "trochoid", noise)
                true_pos, _ = K.fixture_pose(spec)
                true_yaw = math.radians(float(spec.get("angle_deg", 0)))
                cx, cy = true_pos[0] + noise[0], true_pos[1] + noise[1]
                pts = top_points(cx, cy, true_pos[2] + 0.8, true_pos[2] - 0.05,
                                 REG_N, HALF, rig)
                prior = true_yaw + noise[2]
                pitch = 2 * HALF / (REG_N - 1)
                n_ok_count = pts.shape[0] >= 30
                pca_y, pca_degen = pca_yaw(pts, prior)
                mr_y = minrect_yaw(pts, prior, pitch)
                # HONEST plan-yaw error: what the plan is BUILT with, not the residual.
                pca_plan = abs(math.degrees(wrap(pca_y - true_yaw))) if pca_y is not None \
                    else abs(math.degrees(wrap(prior - true_yaw)))
                mr_plan = abs(math.degrees(wrap(mr_y - true_yaw))) if mr_y is not None \
                    else abs(math.degrees(wrap(prior - true_yaw)))
                rows.append(dict(seed=seed, n_pts=int(pts.shape[0]), n_ok_count=n_ok_count,
                                 pca_degen=pca_degen, pca_plan=pca_plan, mr_plan=mr_plan,
                                 mr_none=mr_y is None, prior_err=abs(math.degrees(noise[2])),
                                 elong=spec.get("tank_shape") == "elongated"))
                # every rig in this loop shares physics client 0, so the world accumulates
                # bodies unless it is cleared (pybullet ids are not reused)
                for bid in reversed(range(rig.p.getNumBodies())):
                    rig.p.removeBody(bid)
        pa = np.array([r["pca_plan"] for r in rows])
        ma = np.array([r["mr_plan"] for r in rows])
        out[f"{sig},{sigdeg}"] = dict(
            seeds=len(rows),
            pca_degen_rate=round(float(np.mean([r["pca_degen"] for r in rows])), 4),
            count_test_fail=round(float(np.mean([not r["n_ok_count"] for r in rows])), 4),
            minrect_none_rate=round(float(np.mean([r["mr_none"] for r in rows])), 4),
            pca_plan_med=round(float(np.median(pa)), 2),
            pca_plan_p90=round(float(np.percentile(pa, 90)), 2),
            mr_plan_med=round(float(np.median(ma)), 2),
            mr_plan_p90=round(float(np.percentile(ma, 90)), 2),
            prior_err_med=round(float(np.median([r["prior_err"] for r in rows])), 2),
            n_pts_min=int(min(r["n_pts"] for r in rows)))
        print(out[f"{sig},{sigdeg}"])
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "N191_probe.json"),
              "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
