"""N205 -- is the force-violation rate set by the path's REVERSALS (N202.3) or by the FIXTURE?

Purpose: N202.3 claimed the champion's force-violation rate is a reversal-count artifact
(`pearson(path_len_m, fn_p95) = 0.912` pooled over fixture_A+fixture_R). Reversal count is a
DETERMINISTIC function of (path_mode, fixture geometry), so it is CONSTANT inside a suite: that
0.912 is a between-suite correlation and cannot separate "reversals" from "which fixture it is".
This script (a) counts the reversals the planner actually emits per mode/fixture, and (b) does the
paired 20-seed physics comparison on the force side for four path modes that differ in reversal
geometry. Discriminator: `raster` and `rounded` share row count, reversal count and arclength and
differ ONLY in whether the 180 deg flip is a C0 jump or a C1 semicircle; `trochoid` adds continuous
loop reversals; `fitro` uses the footprint-aware row plan.
Inputs: the rig artifacts under results/aegis_v2 (read-only). Outputs: printed tables + asserts.
No physics is simulated here and no coverage/success value is computed, derived or adjusted: the
rig is the only source of coverage_cont/success/force_compliance. Reversal counts are PLANNER
geometry (deterministic output of scrub_uv), labelled as such.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import kaggle_aegis_sweep as rig  # noqa: E402

ARTS = {
    "r351a": "results/aegis_v2/N205_r351_rounded_vs_raster.jsonl",
    "r351b": "results/aegis_v2/N205_r351_fitro_vs_trochoid.jsonl",
    "r350": "results/aegis_v2/Run350_champion_health_r350.jsonl",
}
SUITES = ("fixture_A", "fixture_B", "fixture_R")
FIELDS = ("force_compliance", "fn_p95", "fn_mean", "coverage_cont", "slip_m")


def episodes(path: str) -> dict:
    """Purpose: index one rig artifact by (suite, seed, path_mode) -> episode record.
    The artifact carries BOTH arms of the paired compare, so path_mode is part of the key --
    without it the baseline arm silently overwrites the candidate.
    Inputs: artifact path. Outputs: {(suite, seed, mode): record}.
    """
    out = {}
    with open(os.path.join(ROOT, path)) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if "coverage_cont" in r and "pts_source" in r and "suite" in r:
                out[(r["suite"], r["seed"], r["path_mode"])] = r
    return out


def geometry(spec: dict, mode: str) -> dict:
    """Purpose: planner-derived reversal geometry of one path mode (no physics).
    Inputs: fixture spec, path mode. Outputs: arclength, hard (>90 deg) reversals, tangent-x
    sign flips, max single-step turn (deg), reversal density per metre.
    """
    pts = rig.scrub_uv(spec, mode)
    p = np.asarray(pts, dtype=float)
    d = np.diff(p, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    keep = seg > 1e-12
    d, seg = d[keep], seg[keep]
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.diff(ang)                       # consecutive segments only: no synthetic first turn
    turn = (turn + math.pi) % (2 * math.pi) - math.pi
    # A "reversal" is one lateral-velocity sign change; flips closer than MERGE_M of
    # arclength are the SAME event traversed in two steps (that is how the rig's C0 raster
    # flip is sampled: 90 deg + 90 deg). Counting the cluster, not the raw flip, is what
    # makes `raster` and `rounded` comparable at equal reversal count.
    MERGE_M = 0.05
    flips_at, arc, events = [], 0.0, 0
    for i in range(1, len(d)):
        arc += float(seg[i])
        if np.sign(d[i - 1, 0]) != np.sign(d[i, 0]) and abs(d[i - 1, 0]) > 1e-12:
            if not flips_at or arc - flips_at[-1] > MERGE_M:
                events += 1
                flips_at.append(arc)
    length = float(seg.sum())
    return {"mode": mode, "arclength_m": round(length, 4), "reversals": events,
            "max_step_turn_deg": round(float(np.degrees(np.abs(turn).max())), 1),
            "reversals_per_m": round(events / length, 2) if length else 0.0}


def paired(a: dict, b: dict, suite: str, field: str,
           ma: str, mb: str) -> dict:
    """Purpose: paired 20-seed difference (mb - ma) on one physics field for one suite.
    Inputs: two episode indexes, suite, field, the two path modes. Outputs: stats dict.
    """
    seeds = sorted({s for (su, s, m) in a if su == suite and m == ma}
                   & {s for (su, s, m) in b if su == suite and m == mb})
    xa = np.asarray([a[(suite, s, ma)][field] for s in seeds], dtype=float)
    xb = np.asarray([b[(suite, s, mb)][field] for s in seeds], dtype=float)
    res = stats.ttest_rel(xb, xa) if len(seeds) > 1 else None
    if not seeds:
        return {"n": 0, "mean_a": float("nan"), "mean_b": float("nan"), "delta": 0.0, "p": None}
    return {"n": len(seeds), "mean_a": round(float(xa.mean()), 4),
            "mean_b": round(float(xb.mean()), 4), "delta": round(float(xb.mean() - xa.mean()), 4),
            "p": float(res[1]) if res is not None else None}


def main() -> int:
    """Purpose: run the audit. Inputs: none. Outputs: 0 if the discriminator holds."""
    r351a, r351b, r350 = (episodes(os.path.join(ROOT, ARTS[k])) for k in ("r351a", "r351b", "r350"))

    # (0) seed determinism: the paired rig is only legitimate if the same seed draws the same
    # friction / tool / customer. r350 and r351b both ran trochoid on the same 20 seeds.
    same = all(abs(r350[k]["friction"] - r351b[k]["friction"]) < 1e-12
               for k in r351b if k in r350) and all(
        abs(r350[k]["coverage_cont"] - r351b[k]["coverage_cont"]) < 1e-12
        for k in r351b if k in r350)
    n_t = sum(1 for k in r351b if k[2] == "trochoid")
    print(f"[determinism] trochoid r350 vs r351b: identical friction AND coverage_cont on "
          f"{n_t} paired episodes: {same}")
    assert same, "rig is not seed-deterministic; paired stats are meaningless"

    # (1) planner-derived reversal geometry -- the manipulated quantity
    print("\n[planner geometry]  (deterministic scrub_uv output, no physics)")
    print(f"{'fixture':10}{'mode':10}{'len_m':>8}{'reversals':>10}{'max_step_turn':>15}"
          f"{'rev/m':>8}")
    geom = {}
    for suite in ("fixture_A", "fixture_B"):
        for mode in ("raster", "rounded", "trochoid", "fitro", "orbit"):
            # orbit is rotation-invariant: it only plans on the round face (the rig refuses
            # an elongated face outright), so it is a fixture_A row only.
            if mode == "orbit" and suite != "fixture_A":
                continue
            g = geometry(rig.FIXTURES[suite], mode)
            geom[(suite, mode)] = g
            print(f"{suite:10}{mode:10}{g['arclength_m']:>8}{g['reversals']:>10}"
                  f"{g['max_step_turn_deg']:>15}{g['reversals_per_m']:>8}")

    # (2) the physics arms, per suite, per mode (physics only, rig-computed)
    arms = [("raster", r350, "raster@r350"), ("raster", r351a, "raster@r351a"),
            ("rounded", r351a, "rounded@r351a"), ("trochoid", r350, "trochoid@r350"),
            ("trochoid", r351b, "trochoid@r351b"), ("fitro", r351b, "fitro@r351b")]
    print("\n[physics, 20 seeds, paired rig]  force_compliance / fn_p95 / coverage_cont / success")
    print(f"{'arm':18}{'fixture':10}{'fcomp':>8}{'fn_p95':>8}{'cov_cont':>10}{'succ':>7}{'len_m':>8}")
    table = {}
    for mode, idx, lab in arms:
        for suite in SUITES:
            eps = [v for k, v in idx.items() if k[0] == suite and k[2] == mode]
            if not eps:
                continue
            row = {f: float(np.mean([e[f] for e in eps])) for f in FIELDS if f in eps[0]}
            succ = float(np.mean([1.0 if e["success"] else 0.0 for e in eps]))
            table[(lab, suite)] = (row["force_compliance"], row.get("fn_p95"),
                                   row["coverage_cont"], succ, eps[0]["path_len_m"])
            print(f"{lab:18}{suite:10}{row['force_compliance']:>8.4f}"
                  f"{row.get('fn_p95', float('nan')):>8.4f}{row['coverage_cont']:>10.4f}"
                  f"{succ:>7.3f}{eps[0]['path_len_m']:>8.4f}")

    # (3) the discriminator: raster vs rounded (same rows, same reversal COUNT, C0 vs C1 flip)
    print("\n[paired delta, rounded - raster]  same rows, same reversal count, C1 vs C0 flip")
    for suite in SUITES:
        for f in ("force_compliance", "fn_p95", "coverage_cont"):
            d = paired(r351a, r351a, suite, f, "raster", "rounded")
            print(f"  {suite:10}{f:18}delta={d['delta']:+.4f}  p={d['p']:.3g}  n={d['n']}")

    # (4) reversal DENSITY differs by ~an order of magnitude between the arms whose force
    # compliance is identical -> reversals are not the driver.
    dens = {m: geom[("fixture_A", m)]["reversals_per_m"]
            for m in ("raster", "rounded", "trochoid", "fitro", "orbit")}
    print("\n[N193 prior run] orbit on fixture_A (rotation-invariant disc sweep, 3.03 m): "
          "force_compliance 0.5507 over 100 seeds at pose_noise 0,0 -- the WORST of all six "
          "arms, with the FEWEST row reversals; its lateral flips are spiral tangent crossings, "
          "not row reversals.")
    fc = {"raster": table[("raster@r351a", "fixture_A")][0],
          "rounded": table[("rounded@r351a", "fixture_A")][0],
          "trochoid": table[("trochoid@r351b", "fixture_A")][0],
          "fitro": table[("fitro@r351b", "fixture_A")][0]}
    fc["orbit"] = 0.5507   # N193_r304_auto_s100_0,0.jsonl (100 seeds, pose_noise 0,0)
    print("\n[fixture_A]  reversal density vs force_compliance (orbit: prior 100-seed N193 run)")
    for m in ("orbit", "fitro", "trochoid", "raster", "rounded"):
        print(f"  {m:10}{dens[m]:>8.2f} rev/m   max_step_turn={geom[('fixture_A', m)]['max_step_turn_deg']:>6.1f} deg"
              f"   fcomp={fc[m]:.4f}")

    # fixture effect: same path, three fixtures
    print("\n[fixture effect]  trochoid, same 20 seeds, three fixtures")
    tro = {s: table[("trochoid@r351b", s)][0] for s in SUITES}
    print("  " + "  ".join(f"{s}={tro[s]:.4f}" for s in SUITES))
    spread_fixture = max(tro.values()) - min(tro.values())
    spread_path = max(fc.values()) - min(fc.values())
    print(f"  fixture-driven spread {spread_fixture:.4f} vs path-driven spread {spread_path:.4f}")

    # Assertions: (i) rounding the reversal does NOT fix the force violation, (ii) the fixture
    # term dominates the path term on fixture_A.
    dA = paired(r351a, r351a, "fixture_A", "force_compliance", "raster", "rounded")
    assert dA["p"] is None or dA["p"] > 0.05 or abs(dA["delta"]) < 0.05, dA
    assert max(abs(table[("raster@r351a", s)][0] - table[("raster@r350", s)][0])
               for s in SUITES) < 1e-9, "r350/r351a raster arms disagree"
    print("\n[paired delta, fitro - trochoid]  footprint-aware row plan vs the champion")
    for suite in SUITES:
        for f in ("force_compliance", "coverage_cont"):
            d = paired(r351b, r351b, suite, f, "trochoid", "fitro")
            print(f"  {suite:10}{f:18}delta={d['delta']:+.4f}  p={d['p']:.3g}  n={d['n']}")
    assert spread_fixture > 3 * max(spread_path, 1e-9), (spread_fixture, spread_path)
    print("\nVERDICT: reversal geometry does not move force_compliance; the fixture term does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())