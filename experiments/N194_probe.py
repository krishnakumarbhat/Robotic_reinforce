#!/usr/bin/env python3
"""N194 probe -- WHY the k=1 arm decays with pose noise and the N192 lattice does not.

Pure GEOMETRY probe of the registration ESTIMATOR (no physics, no rig import, no coverage, no
success). It replicates the rig's exact cast/estimator algebra (experiments/kaggle_aegis_sweep.py
lines 1541-1580): a 32x32 ray grid over the window [-H, H]^2, keep the points whose z is within
1 cm of the max-height mode (one flat top face here, so that test is exact), estimate the centre
as the mean of the kept points. Arm k1 = the single window centred on the NOISY planned centre
(the frozen champion); arm L = the k x k lattice of the SAME window at pitch d = 1.5a, keeping
the cast with the most top points (ties -> prior-nearest, so k=1 is bit-identical to arm k1).

Because the face is a RECTANGLE and the window is AXIS-ALIGNED, the kept rays of one cast are a
contiguous index box I_x x I_y, so no ray casting is needed: for a cast centred at (ox, oy) the
kept x-indices are i with |(-H + ox + i*PITCH)| <= hx, i.e. the integers of
[(-hx + H - ox)/PITCH, (hx + H - ox)/PITCH] intersected with [0, N-1]. The estimator error splits
exactly into
    bias = (exceedance)/2  (a one-sided cut of a rect by an axis-aligned window moves the
                            continuous centroid by half the cut depth)
        + parity term in +-PITCH/2   (the run midpoint sits off the window centre by <= half a
                            ray pitch, phase-random, so it averages to 0 and has fixed scale)
so the CONTINUOUS part scales with the containment exceedance and the DISCRETE part does not.
The lattice picks a cast with exceedance 0, so only the scale-free parity term survives.

Inputs:  none (constants mirror the rig: H, n, and the 0.68 x 0.28 m top face of fixture_B).
Outputs: results/aegis_v2/N194_probe.json -- per sigma the k1 and lattice centroid-error
         median/p90, the truncation fraction, the analytic exceedance term, and the parity
         floor. Contains NO coverage and NO success: it cannot set or move a metric.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

H = 0.6                      # REG_HALF_M in the N192/N194 runs
N = 32                       # REG_N
FACE = (0.34, 0.14)          # top-face half-extents (m) of fixture_B (elongated)
A = H - math.hypot(*FACE)    # single-window containment slack = 0.23230
D = 1.5 * A                  # lattice pitch
PITCH = 2 * H / (N - 1)      # ray pitch on one axis = 38.71 mm
YAW_SIGMA_PER_M = 200.0      # the rig's AEGIS_POSE_NOISE ratio (deg of yaw per m of shift)
SIGMAS = [0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.56, 0.64, 0.80, 1.20, 2.56]
DRAWS = 4000
SEED = 20260930


def kept_run(centre: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous kept index run of one axis for a window centred at `centre` (draws,).

    Inputs: centre (draws,), half = face half-extent along that axis. Outputs: (lo, hi) integer
    index bounds of the kept ray run, clipped to [0, N-1]; hi < lo means the cast misses.
    """
    lo = np.ceil((H - half - centre) / PITCH - 1e-9)
    hi = np.floor((H + half - centre) / PITCH + 1e-9)
    return np.maximum(lo, 0).astype(np.int64), np.minimum(hi, N - 1).astype(np.int64)


def cast_stats(ox: float, oy: float, hx: np.ndarray, hy: np.ndarray,
               ex: np.ndarray, ey: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count and centroid error of one cast, for every draw at once.

    Inputs: cast offset (ox, oy); per-draw face half-extents (hx, hy) and centre errors
    (ex, ey). Outputs: (n_kept, err_x, err_y) per draw. The grid point of index i on the x axis
    is at -H + (ex + ox) + i*PITCH, so the mean of the kept run is its midpoint and the error
    against the TRUE face centre is (that midpoint) - (ex + ox) measured from the window centre.
    """
    lx, hx_i = kept_run(ex + ox, hx)
    ly, hy_i = kept_run(ey + oy, hy)
    n = np.maximum(hx_i - lx + 1, 0) * np.maximum(hy_i - ly + 1, 0)
    # absolute hit coordinates: (centre - H) + i*PITCH; the rig estimates the face centre as
    # their mean and the true face centre is the origin, so the error IS that mean.
    mx = (ex + ox) - H + PITCH * ((lx + hx_i) / 2.0)
    my = (ey + oy) - H + PITCH * ((ly + hy_i) / 2.0)
    return n, np.where(n > 0, mx, np.nan), np.where(n > 0, my, np.nan)


def run(sigma: float, rng: np.random.Generator) -> dict:
    """One sigma: Monte-Carlo the planner-pose error and both registration arms."""
    th = rng.normal(0.0, math.radians(YAW_SIGMA_PER_M * sigma), DRAWS)
    ex, ey = rng.normal(0.0, sigma, (2, DRAWS))
    c, s = np.abs(np.cos(th)), np.abs(np.sin(th))
    hx = FACE[0] * c + FACE[1] * s
    hy = FACE[0] * s + FACE[1] * c
    k = int(math.ceil(4.0 * sigma / A)) + 1 if sigma > A else 1
    offs = [((i - (k - 1) / 2.0) * D, (j - (k - 1) / 2.0) * D) for i in range(k) for j in range(k)]

    n1, x1, y1 = cast_stats(0.0, 0.0, hx, hy, ex, ey)
    best_n = np.zeros(DRAWS, dtype=np.int64)
    best = np.full((DRAWS, 2), np.nan)
    for (ox, oy) in offs:                      # strict '>' keeps the prior-nearest cast on ties
        n, x, y = cast_stats(ox, oy, hx, hy, ex, ey)
        take = n > best_n
        best_n = np.where(take, n, best_n)
        best[take] = np.stack([x[take], y[take]], axis=-1)

    # analytic decomposition of the k1 error: one-sided cut depth of each axis / 2, plus the
    # parity term bounded by half a ray pitch.
    cut_x = np.maximum(0.0, hx + np.abs(ex) - H) / 2.0
    cut_y = np.maximum(0.0, hy + np.abs(ey) - H) / 2.0
    e1 = np.hypot(x1, y1)
    eL = np.hypot(best[:, 0], best[:, 1])
    ok = np.isfinite(e1) & np.isfinite(eL)
    q = lambda v, p: float(np.nanpercentile(v, p))
    return {"sigma_t_m": sigma, "yaw_sigma_deg": YAW_SIGMA_PER_M * sigma, "k": k, "casts": k * k,
            "rays": k * k * N * N, "draws": int(ok.sum()),
            "k1_xy_med_mm": 1000 * q(e1[ok], 50), "k1_xy_p90_mm": 1000 * q(e1[ok], 90),
            "lat_xy_med_mm": 1000 * q(eL[ok], 50), "lat_xy_p90_mm": 1000 * q(eL[ok], 90),
            "k1_cut_med_mm": 1000 * float(np.median(np.hypot(cut_x, cut_y))),
            "k1_cut_frac": float(np.mean((cut_x > 0) | (cut_y > 0))),
            "parity_floor_mm": 1000 * PITCH / 2.0}


def main() -> None:
    """Purpose: run the probe and write its JSON. Inputs: none. Outputs: N194_probe.json."""
    rng = np.random.default_rng(SEED)
    rows = []
    for s in SIGMAS:
        rows.append(run(s, rng))
        r = rows[-1]
        print(f"sigma {r['sigma_t_m']:5.2f} k {r['k']:3d} casts {r['casts']:5d} | "
              f"k1 med {r['k1_xy_med_mm']:7.2f} p90 {r['k1_xy_p90_mm']:7.2f} | "
              f"lat med {r['lat_xy_med_mm']:6.2f} p90 {r['lat_xy_p90_mm']:6.2f} | "
              f"k1 cut med {r['k1_cut_med_mm']:7.2f} mm frac {r['k1_cut_frac']:.3f} | "
              f"parity floor {r['parity_floor_mm']:.2f} mm")
    out = {"probe": "N194 estimator geometry (no physics, no coverage, no success)",
           "constants": {"H_m": H, "n": N, "face_half_m": list(FACE), "a_slack_m": A,
                         "d_pitch_m": D, "ray_pitch_m": PITCH,
                         "yaw_sigma_deg_per_m": YAW_SIGMA_PER_M, "draws": DRAWS, "seed": SEED},
           "rows": rows}
    pathlib.Path("results/aegis_v2/N194_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
