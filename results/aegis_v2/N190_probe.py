"""Purpose: N190 diagnostic -- is the 32x32+MEAN registration estimator the binding floor on
the pose-noise robustness ceiling? Re-creates the rig's exact estimator, with the rig's exact
POSE_NOISE applied, and compares (a) the frozen MEAN estimator and (b) a prior-yaw-frame EXTENT
midpoint, across sigma. Diagnostic only: no keep/discard may cite this file; the decider run
must come from experiments/kaggle_aegis_sweep.py.
Inputs: none. Outputs: results/aegis_v2/N190_probe.json + stdout lines.
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

GRIDS = ((32, 0.35), (128, 0.35), (128, 0.60), (181, 0.60))
SIGMAS = (("0.03", "6"), ("0.06", "12"), ("0.10", "20"), ("0.16", "32"))


def cast(frm, to, rig, cap=1024):
    """Purpose: rayTestBatch in <=cap chunks (pybullet hard-caps one batch at 1024 rays).
    Inputs: start/end lists, rig, cap. Outputs: concatenated result tuples.
    """
    res = []
    for i in range(0, len(frm), cap):
        res.extend(rig.p.rayTestBatch(frm[i:i + cap], to[i:i + cap], physicsClientId=0))
    return res


def top_points(cx, cy, z_hi, z_lo, n, half, rig):
    """Purpose: the rig's exact top-face point set for an n x n grid of half-extent `half`.
    Inputs: grid centre, z bracket, resolution, half extent, rig. Outputs: Nx2 (x, y) array.
    """
    xs = np.linspace(cx - half, cx + half, n)
    ys = np.linspace(cy - half, cy + half, n)
    X, Y = np.meshgrid(xs, ys)
    frm = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_hi)], axis=-1).tolist()
    to = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_lo)], axis=-1).tolist()
    res = cast(frm, to, rig)
    hits = [(r[3][0], r[3][1], r[3][2]) for r in res
            if r[0] != -1 and r[0] != rig.tool and r[0] != rig.plane]
    if not hits:
        return np.zeros((0, 2))
    zm = max(h[2] for h in hits)
    return np.array([(h[0], h[1]) for h in hits if abs(h[2] - zm) < 0.01])


def extent_est(pts, yaw):
    """Purpose: extent-midpoint centroid in the PRIOR-YAW frame. The frozen estimator already
    consumes the prior for the PCA pi-ambiguity branch, so this adds no new prior: the prior
    only supplies the rotation in which the extents are measured.
    Inputs: Nx2 hit points, prior yaw (rad). Outputs: (2,) centroid estimate.
    """
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    q = pts @ R.T
    return (0.5 * (q.max(axis=0) + q.min(axis=0))) @ R


def main():
    out = {}
    for suite in ("fixture_A", "fixture_B", "fixture_R"):
        spec_base = K.FIXTURES.get(suite) or {"tank_shape": "round", "surface": "glossy",
                                              "offset_cm": 0, "angle_deg": 0}
        for sigma_t, sigma_y in SIGMAS:
            K.POSE_NOISE = f"{sigma_t},{sigma_y}"
            rows = []
            for seed in range(20):
                spec = (K.sample_customer_fixture(random.Random(seed * 7919 + 1))
                        if suite == "fixture_R" else dict(spec_base))
                noise = K.pose_noise(random.Random(seed * 104729 + 3))
                rig = K.PyBulletScrub(spec, K.sample_friction(random.Random(seed),
                                                             spec.get("surface", "glossy")),
                                      seed % 3, K.T_MAX, "trochoid", noise)
                true_pos, _ = K.fixture_pose(spec)
                yaw = math.radians(float(spec.get("angle_deg", 0)))
                cx, cy = true_pos[0] + noise[0], true_pos[1] + noise[1]
                z_hi, z_lo = true_pos[2] + 0.8, true_pos[2] - 0.05
                truth = np.array([true_pos[0], true_pos[1]])
                rec = {"seed": seed, "shape": spec.get("tank_shape"),
                       "yaw_deg": spec.get("angle_deg"), "noise_mm": round(
                           float(np.hypot(*noise[:2])) * 1000, 2)}
                for n, half in GRIDS:
                    top = top_points(cx, cy, z_hi, z_lo, n, half, rig)
                    key = f"{n}_{half}"
                    if len(top) < 30:
                        rec[key] = None
                        continue
                    rec[key] = round(float(np.linalg.norm(top.mean(axis=0) - truth)) * 1000, 3)
                    rec[key + "_x"] = round(float(np.linalg.norm(
                        extent_est(top, yaw + noise[2]) - truth)) * 1000, 3)
                rig.close()
                rows.append(rec)
            agg = {}
            for kk in rows[0]:
                if kk[0].isdigit():
                    v = np.array([r[kk] for r in rows if r.get(kk) is not None], dtype=float)
                    agg[kk] = {"med": round(float(np.median(v)), 3),
                               "p90": round(float(np.percentile(v, 90)), 3),
                               "max": round(float(v.max()), 3), "n": int(v.size)}
            out[f"{suite}@{sigma_t}"] = {"mm": agg, "npts_mean": round(
                float(np.mean([r["noise_mm"] for r in rows])), 2)}
            print(suite, sigma_t, json.dumps(agg), flush=True)
    K.POSE_NOISE = "0,0"
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "N190_probe.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
