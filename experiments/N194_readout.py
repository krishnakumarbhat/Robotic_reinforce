#!/usr/bin/env python3
"""N194 readout -- the fixture_B pose-noise frontier of the N192 cast lattice.

Reads the 20-seed paired B scans (lattice dose vs frozen k=1, SAME seeds) plus the 100-seed
3-suite decider at 0.64 m / 128 deg, and writes the frontier table + the frozen-arm 0.70
crossing (logistic fit with a seed bootstrap) to results/aegis_v2/N194_frontier.json.

Pure readout: it re-aggregates numbers the rig already wrote. It computes no coverage and no
success of its own (G7).
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import random
import statistics

SCANS = sorted(glob.glob("results/aegis_v2/N194_r305_s20_B_*.jsonl"),
               key=lambda p: float(p.rsplit("_", 1)[-1].split(",")[0]))
DECIDER = "results/aegis_v2/N194_r305_s100_0.64,128.jsonl"


def episodes(path: str) -> tuple[list[dict], list[dict], dict]:
    """Split a paired rig file into (dose arm, frozen k=1 arm, compare record)."""
    rows = [json.loads(l) for l in open(path)]
    eps = [r for r in rows if r.get("record") not in ("header", "summary", "compare")
           and r.get("suite") == "fixture_B"]
    n = len(eps) // 2
    cmp_rec = next(r for r in rows if r.get("record") == "compare")
    return eps[:n], eps[n:], cmp_rec


def logistic_crossing(sigmas: list[float], succ: list[float], draws: int = 4000) -> dict:
    """sigma where a logistic fit of the frozen arm's success crosses 0.70, bootstrapped."""
    xs = [math.log(s) for s in sigmas]
    ys = [math.log(max(min(p, 1 - 1e-6), 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6))) for p in succ]

    def solve(a: float, b: float) -> float:
        return math.exp(-(a + b * math.log(0.7 / 0.3)) / b) if b != 0 else float("nan")

    def fit(px: list[float], py: list[float]) -> tuple[float, float]:
        b = sum((x - statistics.mean(px)) * (y - statistics.mean(py)) for x, y in zip(px, py))
        a = statistics.mean(py) - b * statistics.mean(px)
        return a, b

    a, b = fit(xs, ys)
    rng = random.Random(20260930)
    # restrict to the resolvable band: at sigma >= 0.96 the frozen arm sits at or below 0.20 and
    # the logit runs to -13, which drags the linear-in-log-sigma slope and biases the crossing low.
    keep = [i for i, p in enumerate(succ) if 0.2 <= p <= 0.95]
    a2, b2 = fit([xs[i] for i in keep], [ys[i] for i in keep])
    boots = []
    for _ in range(draws):
        idx = [keep[rng.randrange(len(keep))] for _ in keep]
        ab, bb = fit([xs[i] for i in idx], [ys[i] for i in idx])
        if bb < 0:
            boots.append(solve(ab, bb))
    boots.sort()
    below = [s for s, p in zip(sigmas, succ) if p <= 0.70]
    above = [s for s, p in zip(sigmas, succ) if p > 0.70]
    return {"fit_all_sigma_m": solve(a, b), "slope_logit_per_ln_sigma": b,
            "fit_resolvable_band_m": solve(a2, b2), "band_used_sigma_m": [sigmas[i] for i in keep],
            "empirical_bracket_m": [max(above) if above else None,
                                    min(below) if below else None],
            "boot_p2.5": boots[int(0.025 * len(boots))], "boot_p97.5": boots[int(0.975 * len(boots))],
            "boot_median": boots[len(boots) // 2]}


def main() -> None:
    """Purpose: build the frontier table. Inputs: the rig JSONLs. Outputs: N194_frontier.json."""
    rows, sig, fsucc = [], [], []
    for p in SCANS:
        dose, frz, c = episodes(p)
        s = c["pose_noise_cfg"]
        row = {"pose_noise_cfg": s, "sigma_t_m": float(s.split(",")[0]),
               "casts": dose[0]["reg_casts"],
               "dose_covc": c["fixture_B"]["mean_b"], "dose_success": c["fixture_B"]["succ_b"],
               "frozen_covc": c["fixture_B"]["mean_a"], "frozen_success": c["fixture_B"]["succ_a"],
               "welch_p": c["fixture_B"]["welch_p"], "fisher_p": c["fixture_B"]["fisher_p"],
               "keep": c["keep"], "n_seeds": len(dose),
               "dose_reg_ok": sum(1 for e in dose if e.get("reg_ok")),
               "dose_xy_med_mm": 1000 * statistics.median([e["reg_err_xy_m"] for e in dose
                                                           if e.get("reg_err_xy_m") is not None]),
               "frozen_xy_med_mm": 1000 * statistics.median([e["reg_err_xy_m"] for e in frz
                                                             if e.get("reg_err_xy_m") is not None]),
               "frozen_reg_ok": sum(1 for e in frz if e.get("reg_ok"))}
        rows.append(row)
        sig.append(row["sigma_t_m"])
        fsucc.append(row["frozen_success"] / row["n_seeds"])
    drows = [json.loads(l) for l in open(DECIDER)]
    cmp_rec = next(r for r in drows if r.get("record") == "compare")
    out = {"idea": "N194", "question": "where does fixture_B itself fall below the 0.70 bar?",
           "arms": {"candidate": "trochoid + N192 cast lattice (AEGIS_REG_CASTS=0)",
                    "baseline": "trochoid + frozen single cast (AEGIS_REG_CASTS=1)"},
           "frontier_scans_fixture_B": rows,
           "frozen_arm_success": dict(zip(sig, fsucc)),
           "frozen_arm_crossing": logistic_crossing(sig, fsucc),
           "decider_100_seeds": {"pose_noise_cfg": cmp_rec["pose_noise_cfg"],
                                 "per_suite": {k: v for k, v in cmp_rec.items()
                                               if k.startswith("fixture_")},
                                 "keep": cmp_rec["keep"]},
           "not_finished": {"pose_noise_cfg": "3.20,640", "k": 57, "casts": 3249,
                            "rays_per_episode": 3249 * 1024,
                            "why": "20 seeds x 2 arms exceeded the 120 s wall budget "
                                   "(~9 s/episode at 2116 casts, linear in casts)"}}
    p = pathlib.Path("results/aegis_v2/N194_frontier.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"{'sigma':>6} {'casts':>6} {'dose covc':>10} {'dose s':>7} {'frz covc':>9} "
          f"{'frz s':>6} {'dose ok':>8} {'frz ok':>7} {'fisher p':>10} {'keep':>5}")
    for r in rows:
        print(f"{r['sigma_t_m']:6.2f} {r['casts']:6d} {r['dose_covc']:10.4f} "
              f"{r['dose_success']:3d}/{r['n_seeds']:<3d} {r['frozen_covc']:9.4f} "
              f"{r['frozen_success']:3d}/{r['n_seeds']:<3d} {r['dose_reg_ok']:4d}/{r['n_seeds']:<3d} "
              f"{r['frozen_reg_ok']:3d}/{r['n_seeds']:<3d} {r['fisher_p']:10.2e} {str(r['keep']):>5}")
    print("frozen-arm 0.70 crossing (logistic on log sigma, seed bootstrap):",
          json.dumps(out["frozen_arm_crossing"]))
    print("decider 100 seeds:", json.dumps(out["decider_100_seeds"]["per_suite"]["fixture_B"]),
          "keep =", out["decider_100_seeds"]["keep"])


if __name__ == "__main__":
    main()
