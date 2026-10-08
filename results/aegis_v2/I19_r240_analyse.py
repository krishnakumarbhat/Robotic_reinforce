#!/usr/bin/env python3
"""Purpose: read-out for I19 (advisory slowdown gate, run 240).

Inputs:  argv[1] = the paired run jsonl (candidate arm + paired baseline arm).
Outputs: per-suite coverage_cont / success / jerk / slip / fn_p95 tables for both arms,
         the gate telemetry (vetoes, vetoed_ticks, jerk_tick_max) and the Welch/Fisher
         comparisons the rig already computed, printed as one dense report.

Nothing here computes or adjusts a scored quantity: every number is read from the rig
rows (G7). Paired analysis (same seeds) is printed alongside the rig's own compare record.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import fisher_exact, ttest_ind, wilcoxon


def rows(path: str) -> list[dict]:
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("record") == "episode" and str(r.get("status", "")).startswith("PHYSICAL"):
            out.append(r)
    return out


def main() -> int:
    recs = rows(sys.argv[1])
    header = json.loads(open(sys.argv[1]).readline())
    arms: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        # arms are distinguished by the gate flag the arm actually ran with: the candidate
        # arm has AEGIS_GATE_ADVISORY=1 (from the shell env), the paired baseline arm had
        # the knob overridden to 0. Path mode is identical in both (same --path).
        arms["candidate" if r.get("gate_on") else "baseline"].append(r)
    print(f"pose_noise_cfg={header.get('pose_noise_cfg')} gate_mode={header.get('gate_mode')}"
          f" path={header.get('path_mode')} seeds={header.get('seeds_requested')}")
    for arm in ("candidate", "baseline"):
        for suite in ("fixture_A", "fixture_B", "fixture_R"):
            rs = [r for r in arms[arm] if r["suite"] == suite]
            if not rs:
                continue
            cov = np.array([r["coverage_cont"] for r in rs])
            suc = np.array([r["success"] for r in rs])
            jk = np.array([r["jerk"] for r in rs])
            sl = np.array([r["slip_m"] for r in rs])
            p95 = np.array([r.get("fn_p95", 0.0) for r in rs])
            vo = np.array([r.get("vetoes", 0) for r in rs])
            vt = np.array([r.get("vetoed_ticks", 0) for r in rs])
            jm = np.array([r.get("jerk_tick_max", 0.0) for r in rs])
            ln = np.array([r.get("path_len_m", 0.0) for r in rs])
            print(f"  {arm:9s} {suite} n={len(rs):2d} succ={suc.sum():2d}/{len(rs)} "
                  f"covc={cov.mean():.4f}+-{cov.std(ddof=1):.4f} min={cov.min():.4f} "
                  f"jerk={jk.mean():.4f} max={jk.max():.4f} slip90={sl.mean():.4f} "
                  f"fn_p95={p95.mean():.4f} vetoes={vo.sum():3d} slowticks={vt.sum():4d} "
                  f"jmax={jm.max():.3f} len={ln.mean():.4f}")
    for suite in ("fixture_A", "fixture_B", "fixture_R"):
        a = [r["coverage_cont"] for r in arms["baseline"] if r["suite"] == suite]
        b = [r["coverage_cont"] for r in arms["candidate"] if r["suite"] == suite]
        sa = [bool(r["success"]) for r in arms["baseline"] if r["suite"] == suite]
        sb = [bool(r["success"]) for r in arms["candidate"] if r["suite"] == suite]
        if not a or not b:
            continue
        welch = float(ttest_ind(b, a, equal_var=False)[1])
        try:
            paired = float(wilcoxon(b, a)[1]) if len(a) == len(b) else float("nan")
        except ValueError:
            paired = float("nan")
        try:
            fish = float(fisher_exact([[sum(sb), len(sb) - sum(sb)],
                                       [sum(sa), len(sa) - sum(sa)]])[1])
        except Exception:  # noqa: BLE001
            fish = float("nan")
        print(f"  PAIRED {suite}: dcovc={np.mean(b) - np.mean(a):+.4f} welch_p={welch:.3e} "
              f"paired_p={paired:.3e} succ {sum(sa)}/20->{sum(sb)}/20 fisher_p={fish:.3e}")
    for line in open(sys.argv[1]):
        r = json.loads(line)
        if r.get("record") == "compare":
            print("  RIG compare keep=", r.get("keep"))
            for suite in ("fixture_A", "fixture_B", "fixture_R"):
                if suite in r:
                    c = r[suite]
                    print(f"    {suite}: delta={c.get('delta')} welch_p={c.get('welch_p')} "
                          f"fisher_p={c.get('fisher_p')} succ {c.get('succ_a')}->{c.get('succ_b')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
