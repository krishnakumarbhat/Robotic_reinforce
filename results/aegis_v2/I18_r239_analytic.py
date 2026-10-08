"""I18 pre-count check: does the trochoid-loop x fitted-pitch COMPOSITION buy anything?

Purpose: `fitted` already attains the analytic coverage ceiling 1.0000 on all six
    suite x tool cells (I21 run 238), so the composition `fitro` can only win on a
    NON-coverage channel (cycle time / physics robustness). Before spending the single
    20-seed paired run, price that channel in closed form against the rig's own scoring
    kernel: ceiling, path length, C1 turn and the loop's marginal arclength cost.
    No physics, no rig edit beyond adding the `fitro` path mode.
Inputs: none (imports the frozen rig module + I21's ceiling kernel).
Outputs: results/aegis_v2/I18_r239_analytic.json + a printed table.
"""
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT)
import kaggle_aegis_sweep as rig  # noqa: E402

_spec = importlib.util.spec_from_file_location("i21_analytic", os.path.join(HERE, "I21_r238_analytic.py"))
i21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i21)          # reuse the SAME fine-cell kernel, not a re-derived one

FIXTURES = {"fixture_A": {"tank_shape": "round", "surface": "glossy", "offset_cm": 0, "angle_deg": 0},
            "fixture_B": {"tank_shape": "elongated", "surface": "matte", "offset_cm": 15, "angle_deg": 10}}
TOOLS = {0: [0.05, 0.035, 0.012], 1: [0.04, 0.04, 0.030], 2: [0.09, 0.05, 0.006]}
MODES = ("trochoid", "fitted", "fitro")


def main():
    out = {"rig_version": 2, "note": "analytic pre-count: composition priced in closed form "
            "with the rig's own fine-cell kernel; physics can only subtract from the ceiling",
           "modes": {}, "verdict_inputs": {}}
    print(f"{'suite':10s} {'tool':4s} {'mode':9s} {'ceil':7s} {'len_m':7s} {'turn':6s} {'npts':6s}")
    for sname, spec in FIXTURES.items():
        for tid, half in TOOLS.items():
            r_eff = float(min(half[0], half[1]))
            for mode in MODES:
                cov, hit = i21.ceiling(spec, mode, r_eff)
                uv = rig.scrub_uv(dict(spec, r_eff=r_eff), mode)
                L, turn = rig.uv_length(uv), rig.max_turn_deg(uv)
                prof = i21.miss_profile(hit, spec)
                key = f"{sname}/tool{tid}/{mode}"
                out["modes"][key] = {"ceiling": round(cov, 4), "path_len_m": round(L, 4),
                                     "max_turn_deg": round(turn, 3), "n_points": len(uv),
                                     "r_eff": r_eff, **prof}
                print(f"{sname:10s} {tid:<4d} {mode:9s} {cov:<7.4f} {L:<7.4f} {turn:<6.2f} {len(uv):<6d}")

    # the pre-registered question: relative to `fitted` (the coverage-saturated arm), what
    # does the composition COST in arclength, and can it beat trochoid on cycle time?
    v = {}
    for sname in FIXTURES:
        g = lambda m: [x for k, x in out["modes"].items() if k.startswith(sname) and k.endswith("/" + m)]
        f, t, c = g("fitted"), g("trochoid"), g("fitro")
        v[sname] = {
            "ceiling": {m: [y["ceiling"] for y in g(m)] for m in MODES},
            "len_mean_m": {m: round(sum(y["path_len_m"] for y in g(m)) / 3, 4) for m in MODES},
            "fitro_vs_fitted_len_ratio": round(
                sum(y["path_len_m"] for y in c) / sum(y["path_len_m"] for y in f), 4),
            "fitro_vs_trochoid_len_ratio": round(
                sum(y["path_len_m"] for y in c) / sum(y["path_len_m"] for y in t), 4),
            "fitro_ceiling_minus_fitted": round(
                sum(y["ceiling"] for y in c) / 3 - sum(y["ceiling"] for y in f) / 3, 4),
            "fitro_ceiling_minus_trochoid": round(
                sum(y["ceiling"] for y in c) / 3 - sum(y["ceiling"] for y in t) / 3, 4),
            "fitro_turn_max_deg": max(y["max_turn_deg"] for y in c),
        }
    out["verdict_inputs"] = v
    print("\ncomposition vs its parts (mean over the 3 pads):")
    for s, x in v.items():
        print(f"  {s}: ceil troch {x['ceiling']['trochoid']} -> fitro {x['ceiling']['fitro']} "
              f"(d_fitted {x['fitro_ceiling_minus_fitted']:+.4f}, d_trochoid {x['fitro_ceiling_minus_trochoid']:+.4f})")
        print(f"      len fitted {x['len_mean_m']['fitted']:.4f} m, fitro {x['len_mean_m']['fitro']:.4f} m "
              f"(x{x['fitro_vs_fitted_len_ratio']:.4f} vs fitted, x{x['fitro_vs_trochoid_len_ratio']:.4f} vs trochoid), "
              f"max turn {x['fitro_turn_max_deg']:.2f} deg (C1 bar {rig.MAX_TURN_DEG})")
    with open(os.path.join(HERE, "I18_r239_analytic.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'I18_r239_analytic.json')}")


if __name__ == "__main__":
    main()
