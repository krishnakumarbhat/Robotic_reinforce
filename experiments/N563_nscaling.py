"""Read-only: log-log fit of reg_err_xy p50 against REG_N for the two estimators.

Uses ONLY the four archived N563b cells (REG_N = 16/32/64/128 at pose_noise 3.20,640, 20 seeds,
paired). reg_err_xy is read straight out of each rig episode record; the fit is an ordinary
least-squares line in (log N, log p50). No coverage or success quantity is touched.
"""
import numpy as np

import importlib.util

spec = importlib.util.spec_from_file_location("s", "experiments/N563_summarize.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

NS = [16, 32, 64, 128]
CELLS = {n: f"N563b_N{n}" for n in NS}
# N=16 is excluded from the fit: fixture_B/fixture_R ESCAPE there (16 of 20 B episodes escaped,
# reg_ok 4/20), so it is off the truncation-free branch and its reg_err is not the same statistic.
FIT_NS = [32, 64, 128]

if __name__ == "__main__":
    data = {}
    for n, tag in CELLS.items():
        c = S.cell(tag)
        if c is None:
            print(f"{tag}: MISSING")
            continue
        for arm_lbl in ("mean", "extent"):
            for su in ("fixture_A", "fixture_B", "fixture_R"):
                pre = "base[" if arm_lbl == "mean" else "cand["
                k = next(k for k in c["per"] if k.startswith(pre) and k.endswith("/" + su))
                v = c["per"][k]["reg_err_p50_mm"]
                data.setdefault((su, arm_lbl), {})[n] = v
    print(f"{'suite':10s} {'estimator':10s} " + " ".join(f"{'N=' + str(n):>9s}" for n in NS)
          + "   slope(N=32..128)  ratio base/extent @N=128")
    out = {}
    for su in ("fixture_A", "fixture_B", "fixture_R"):
        for arm in ("mean", "extent"):
            d = data.get((su, arm), {})
            xs = np.array([np.log(n) for n in FIT_NS])
            ys = np.array([d[n] for n in FIT_NS])
            slope, icpt = np.polyfit(xs, ys, 1)
            pred = slope * xs + icpt
            ss_res = float(((ys - pred) ** 2).sum())
            ss_tot = float(((ys - ys.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rat = (data[(su, "mean")][128] / data[(su, "extent")][128]
                   if data.get((su, "mean"), {}).get(128) and data[(su, "extent")].get(128)
                   else float("nan"))
            out[f"{su}/{arm}"] = {"p50_mm": {str(k): v for k, v in d.items()},
                                  "slope_logN": round(float(slope), 4),
                                  "r2": round(r2, 5),
                                  "ratio_mean_over_extent_at_128": round(float(rat), 4)}
            print(f"{su:10s} {arm:10s} "
                  + " ".join(f"{d[n]:9.4f}" if d.get(n) is not None else f"{'-':>9s}" for n in NS)
                  + f"   {slope:+.4f} (R2 {r2:.4f})   {rat:.4f}")
    import json
    print(json.dumps(out, indent=1))
