#!/usr/bin/env python3
"""N196 probe -- is the PARITY term 2H/(n-1) still VISIBLE? Sweep the ray resolution n.

Pure GEOMETRY probe of the registration ESTIMATOR (no physics, no rig import, no coverage, no
success). It replicates the rig's exact cast/estimator algebra: an n x n ray grid over the window
[-H, H]^2, keep the points within 1 cm of the max-height mode (one flat top face, so exact),
estimate the centre as the mean of the kept points. Fixture_B's top face is a RECTANGLE and the
window is AXIS-ALIGNED, so the kept rays of one cast are a contiguous index box I_x x I_y and no
ray casting is needed: for a cast centred at (ox, oy) the kept x-indices are i with
|(-H + ox + e_x + i*PITCH)| <= hx, i.e. the integers of [(-hx + H - ox - e_x)/PITCH,
(hx + H - ox - e_x)/PITCH] intersected with [0, n-1].

N194.1 showed the estimator error is `d_cut/2 + eps_parity` and the N192/N195 lattice forces
d_cut = 0, so ONLY eps_parity survives, with PITCH = 2H/(n-1). N196 asks the one question the
N194->N196 edge names: is that surviving term still measurable in coverage_cont, i.e. does
halving/quartering PITCH buy anything? The probe answers the estimator half of it exactly:
sweep n in {16, 32, 64, 128} and report the parity amplitude, the realised centroid error, and
whether the cast is even resolvable (rig's own reg_ok gate: >= 30 top points).

It predicts the direction and size of the rig pairing; it CANNOT set or move a metric.

Inputs:  none (constants mirror the rig: H, and the 0.68 x 0.28 m top face of fixture_B).
Outputs: results/aegis_v2/N196_probe.json -- per (n, sigma) the parity amplitude, the lattice-arm
         centroid-error median/p90/max, kept-point counts and the reg_ok rate.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

H = 0.6                      # REG_HALF_M in the N192/N194/N195 runs
FACE = (0.34, 0.14)          # top-face half-extents (m) of fixture_B (elongated)
A = H - math.hypot(*FACE)    # single-window containment slack = 0.23230
D = 1.5 * A                  # lattice pitch (the SHIPPED d = 1.5a; N195 kept it exact)
YAW_SIGMA_PER_M = 200.0      # the rig's AEGIS_POSE_NOISE ratio (deg of yaw per m of shift)
NS = [16, 32, 64, 128]       # ray resolutions: shipped n=32, and the untested n=64/128
SIGMAS = [0.28, 0.64, 3.20]  # 0.28 the run-303 level, 0.64 the N195 level, 3.20 the r306 level
REG_OK_MIN = 30              # the rig's reg_ok gate on the number of top-face points
DRAWS = 4000
SEED = 20260930


def kept_run(centre: np.ndarray, half: np.ndarray, n: int,
             pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous kept index run of one axis, for every draw at once.

    Inputs: centre (draws,) cast centre along that axis, half = face half-extent, n, pitch.
    Outputs: (lo, hi) integer bounds of the kept ray run clipped to [0, n-1]; hi < lo = miss.
    """
    lo = np.ceil((H - half - centre) / pitch - 1e-9)
    hi = np.floor((H + half - centre) / pitch + 1e-9)
    return np.maximum(lo, 0).astype(np.int64), np.minimum(hi, n - 1).astype(np.int64)


def cast_stats(ox: float, oy: float, hx: np.ndarray, hy: np.ndarray,
               ex: np.ndarray, ey: np.ndarray, n: int, pitch: float) -> tuple:
    """Count and centroid error of one cast, for every draw at once.

    Inputs: cast offset (ox, oy); per-draw face half-extents (hx, hy) and centre errors
    (ex, ey); n, pitch. Outputs: (n_kept, err_x, err_y) per draw. The grid point of index i on
    the x axis sits at -H + (ex + ox) + i*pitch, so the mean of the kept run is its midpoint and
    the rig's estimate minus the true face centre (origin) IS that mean.
    """
    lx, hx_i = kept_run(ex + ox, hx, n, pitch)
    ly, hy_i = kept_run(ey + oy, hy, n, pitch)
    cnt = np.maximum(hx_i - lx + 1, 0) * np.maximum(hy_i - ly + 1, 0)
    mx = (ex + ox) - H + pitch * ((lx + hx_i) / 2.0)
    my = (ey + oy) - H + pitch * ((ly + hy_i) / 2.0)
    return cnt, np.where(cnt > 0, mx, np.nan), np.where(cnt > 0, my, np.nan)


def run(n: int, sigma: float, rng: np.random.Generator) -> dict:
    """One (n, sigma): Monte-Carlo the planner-pose error and the N195 lattice arm.

    Inputs: n ray resolution, sigma_t in metres, seeded generator. Outputs: the parity readout
    dict. The arm is the SHIPPED one: k = ceil(6 sigma/d)+1 offsets at pitch d, argmax on the
    border margin (N195.2), one fine cast at n for the winner.
    """
    pitch = 2 * H / (n - 1)
    th = rng.normal(0.0, math.radians(YAW_SIGMA_PER_M * sigma), DRAWS)
    ex, ey = rng.normal(0.0, sigma, (2, DRAWS))
    c, s = np.abs(np.cos(th)), np.abs(np.sin(th))
    hx = FACE[0] * c + FACE[1] * s
    hy = FACE[0] * s + FACE[1] * c
    k = (int(math.ceil(6.0 * sigma / D)) + 1) if sigma > A else 1
    offs = [((i - (k - 1) / 2.0) * D, (j - (k - 1) / 2.0) * D) for i in range(k) for j in range(k)]

    def _margin(ox, oy, cnt):
        """Border margin in metres: smallest gap from the point bbox to the window border."""
        lx, hx_i = kept_run(ex + ox, hx, n, pitch)
        ly, hy_i = kept_run(ey + oy, hy, n, pitch)
        x_lo, x_hi = ex + ox - H, ex + ox + H
        y_lo, y_hi = ey + oy - H, ey + oy + H
        gx = np.minimum((lx) * pitch, (n - 1 - hx_i) * pitch)
        gy = np.minimum((ly) * pitch, (n - 1 - hy_i) * pitch)
        return np.where(cnt > 0, np.minimum(gx, gy), -1.0)

    best_m = np.full(DRAWS, -1.0)
    best_off = np.zeros((DRAWS, 2))
    for (ox, oy) in offs:
        cnt, _, _ = cast_stats(ox, oy, hx, hy, ex, ey, n, pitch)
        m = _margin(ox, oy, cnt)
        take = m > best_m                      # strict: ties keep the prior-nearest cast
        best_m = np.where(take, m, best_m)
        best_off[take] = (ox, oy)
    cnt = np.zeros(DRAWS, dtype=np.int64)
    err = np.full((DRAWS, 2), np.nan)
    for (ox, oy) in offs:
        sel = (best_off[:, 0] == ox) & (best_off[:, 1] == oy)
        if not sel.any():
            continue
        idx = np.where(sel)[0]
        c_i, x_i, y_i = cast_stats(ox, oy, hx[idx], hy[idx], ex[idx], ey[idx], n, pitch)
        cnt[idx] = c_i
        err[idx, 0] = x_i
        err[idx, 1] = y_i

    e = np.hypot(err[:, 0], err[:, 1])
    ok = np.isfinite(e) & (cnt >= REG_OK_MIN)
    q = lambda v, p: float(np.nanpercentile(v, p))          # noqa: E731
    return {"n": n, "sigma_t_m": sigma, "k": k, "casts": k * k, "rays": k * k * n * n,
            "ray_pitch_mm": 1000 * pitch, "parity_amp_mm": 1000 * pitch / 2.0,
            "parity_rms_mm": 1000 * pitch / np.sqrt(12.0),
            "reg_ok_rate": float(ok.mean()), "kept_med": float(np.median(cnt)),
            "err_med_mm": 1000 * q(e[ok], 50), "err_p90_mm": 1000 * q(e[ok], 90),
            "err_max_mm": 1000 * float(np.max(e[ok])) if ok.any() else None,
            "err_over_parity_p90": q(e[ok], 90) / (pitch / 2.0) if ok.any() else None}


def main() -> None:
    """Purpose: sweep n x sigma and write the probe JSON. Inputs: none.
    Outputs: results/aegis_v2/N196_probe.json. Contains no coverage and no success.
    """
    rows = []
    for n in NS:
        rng = np.random.default_rng(SEED + n)
        for sig in SIGMAS:
            r = run(n, sig, rng)
            rows.append(r)
            print(f"n {r['n']:4d} sigma {r['sigma_t_m']:5.2f} k {r['k']:4d} "
                  f"rays {r['rays']:9d} | pitch {r['ray_pitch_mm']:6.2f} amp {r['parity_amp_mm']:5.2f} "
                  f"rms {r['parity_rms_mm']:5.2f} mm | kept {r['kept_med']:6.0f} "
                  f"reg_ok {r['reg_ok_rate']:.3f} | err med {r['err_med_mm']:6.2f} "
                  f"p90 {r['err_p90_mm']:6.2f} max {r['err_max_mm'] or -1:7.2f} mm "
                  f"(p90/amp {r['err_over_parity_p90']:.2f})")
    out = {"probe": "N196 parity-vs-resolution geometry (no physics, no coverage, no success)",
           "constants": {"H_m": H, "face_half_m": list(FACE), "a_slack_m": A, "d_pitch_m": D,
                         "yaw_sigma_deg_per_m": YAW_SIGMA_PER_M, "draws": DRAWS, "seed": SEED,
                         "reg_ok_min_points": REG_OK_MIN, "ns": NS, "sigmas": SIGMAS},
           "rows": rows}
    pathlib.Path("results/aegis_v2/N196_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
