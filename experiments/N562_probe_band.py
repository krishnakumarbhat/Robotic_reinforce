#!/usr/bin/env python3
"""N562 geometry probe (READ-ONLY, no coverage, no success, no teleport).

Answers one arithmetic question the N561 diagnostics raised but could not settle from logged
scalars alone: what EXACTLY does the depth-registration 1 cm top-face band accept on
`fixture_B`? It rebuilds ONE world with the rig's own PyBulletScrub (noise = 0), casts the
certified N478 grid (H=0.6, n=32, offsets (0,0)) and prints
  - the accepted band size and its oriented bbox vs the frozen face extents 0.68 x 0.28 m,
  - the WORLD-axis bbox, and
  - the distinct z values and rigid-body ids inside the band (cell (i) of D1).
It reports geometry only: no scrub pass is run, nothing is scored, nothing is written to any
results file. Any disagreement with `results/aegis_v2/N562_*.jsonl` is reported, not hidden.
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec_mod = importlib.util.spec_from_file_location("aegis_rig", os.path.join(HERE, "kaggle_aegis_sweep.py"))
rig = importlib.util.module_from_spec(spec_mod)
sys.argv = ["probe"]
spec_mod.loader.exec_module(rig)

H = 0.6
N = 32
spec = dict(rig.FIXTURES["fixture_B"])
true_pos, _ = rig.fixture_pose(spec)
r = rig.PyBulletScrub(spec, 0.4, 0, rig.T_MAX, "trochoid", (0.0, 0.0, 0.0))
p = r.p
cx, cy = true_pos[0], true_pos[1]
z_hi, z_lo = true_pos[2] + 0.8, true_pos[2] - 0.05
xs = np.linspace(cx - H, cx + H, N)
ys = np.linspace(cy - H, cy + H, N)
X, Y = np.meshgrid(xs, ys)
frm = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(z_hi))], -1).tolist()
to = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(z_lo))], -1).tolist()
res = []
for i in range(0, len(frm), 1024):
    res.extend(p.rayTestBatch(frm[i:i + 1024], to[i:i + 1024], physicsClientId=0))
hits = [(i, q[3][0], q[3][1], q[3][2], q[0]) for i, q in enumerate(res)
        if q[0] != -1 and q[0] != r.tool and q[0] != r.plane]
zmax = max(h[3] for h in hits)
band = [(h[1], h[2], h[3]) for h in hits if abs(h[3] - zmax) < 0.01]
rej = [h for h in hits if abs(h[3] - zmax) >= 0.01]
P = np.array([[b[0], b[1]] for b in band])
pitch = 2.0 * H / (N - 1)
cov = np.cov(P[:, 0], P[:, 1])
ev, evec = np.linalg.eigh(cov)
major = evec[:, 1] if ev[1] >= ev[0] else evec[:, 0]
minor = np.array([-major[1], major[0]])
R = np.array([[major[0], -major[1]], [major[1], major[0]]])
q = P @ R.T
zb = sorted({round(b[2], 6) for b in band})
ids = sorted({h[4] for h in hits if abs(h[3] - zmax) < 0.01})
allids = sorted({h[4] for h in hits})
out = {
    "n_hits": len(hits), "n_band": len(band), "n_rejected": len(rej),
    "pitch_m": round(pitch, 6),
    "band_area_implied_m2": round(len(band) * pitch ** 2, 5),
    "face_area_m2": round(0.68 * 0.28, 5),
    "band_z_span_m": round(max(b[2] for b in band) - min(b[2] for b in band), 8),
    "band_z_values": zb[:8], "n_distinct_band_z": len(zb),
    "band_body_ids": ids, "all_hit_body_ids": allids,
    "world_bbox_du_m": round(float(P[:, 0].max() - P[:, 0].min()), 5),
    "world_bbox_dv_m": round(float(P[:, 1].max() - P[:, 1].min()), 5),
    "oriented_bbox_du_m": round(float(q[:, 0].max() - q[:, 0].min()), 5),
    "oriented_bbox_dv_m": round(float(q[:, 1].max() - q[:, 1].min()), 5),
    "oriented_bbox_du_minus_0.68": round(float(q[:, 0].max() - q[:, 0].min()) - 0.68, 5),
    "oriented_bbox_dv_minus_0.28": round(float(q[:, 1].max() - q[:, 1].min()) - 0.28, 5),
    "pca_ratio": round(float(ev[1] / ev[0]), 4),
    "pca_major_deg": round(math.degrees(math.atan2(major[1], major[0])), 3),
    "true_yaw_deg": float(spec.get("angle_deg", 0)),
    "centroid_err_mm": [round(1000 * float(P[:, 0].mean() - true_pos[0]), 3),
                        round(1000 * float(P[:, 1].mean() - true_pos[1]), 3)],
    "rejected_z_sample": sorted({round(h[3], 4) for h in rej})[:8],
}
# --- direct extent read-out along a fan of directions: the oriented bbox must be the
# MAXIMUM over directions, and for a 0.68 x 0.28 face at 10 deg the extremes are ~0.68 / ~0.28.
fan = {}
for deg in (0.0, 10.0, 20.0, 45.0, 90.0, 100.0, 110.0, 135.0, 169.945, 190.0):
    a = math.radians(deg)
    d = np.array([math.cos(a), math.sin(a)])
    proj = P @ d
    fan[f"{deg:g}deg"] = round(float(proj.max() - proj.min()), 5)
out["extent_by_direction_m"] = fan
out["oriented_frame_q0_range"] = [round(float(q[:, 0].min()), 5), round(float(q[:, 0].max()), 5)]
out["oriented_frame_q1_range"] = [round(float(q[:, 1].min()), 5), round(float(q[:, 1].max()), 5)]
out["world_x_range"] = [round(float(P[:, 0].min()), 5), round(float(P[:, 0].max()), 5)]
out["world_y_range"] = [round(float(P[:, 1].min()), 5), round(float(P[:, 1].max()), 5)]
out["true_pos_xy"] = [round(true_pos[0], 5), round(true_pos[1], 5)]
print(json.dumps(out, indent=1))