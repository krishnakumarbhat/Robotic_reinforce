#!/usr/bin/env python3
"""N196 readout -- one table per paired rig file: coverage_cont, transfer_success, the
registration residual and the wall cost, arm by arm, plus the probe's parity law.

Inputs:  file paths on argv (rig JSONLs written by experiments/kaggle_aegis_sweep.py) and the
         optional N196_probe.json. Outputs: a printed table (stdout only; it writes no metric).
         Every number it prints is read back from the JSONL; nothing is recomputed.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys


def pct(v: list[float], p: float) -> float:
    """Percentile of a non-empty list (0.0 if empty)."""
    return 0.0 if not v else float(sorted(v)[min(len(v) - 1, int(round(p * (len(v) - 1))))])


def arm_rows(rows: list[dict], arm: int, suite: str) -> list[dict]:
    """Physical episode records of one ARM x suite.

    Inputs: rows (the whole file), arm = 0 for the candidate block, 1 for the `--compare-env`
    baseline block, suite. Outputs: that arm's physical records for the suite.

    The rig runs ALL candidate episodes first and then applies `--compare-env` and runs the whole
    baseline block (kaggle_aegis_sweep.py: the outer loop is over modes, the inner over
    suites x seeds), and both arms share `path_mode` when they differ only in a geometry knob --
    so the arm is positional: the first half of each suite's records is the candidate.
    """
    sel = [r for r in rows if r.get("record") not in ("compare", "summary", "summary_mode")
           and r.get("suite") == suite and str(r.get("status", "")).startswith("PHYSICAL")]
    half = len(sel) // 2
    if len(sel) % 2:
        raise SystemExit(f"{suite}: {len(sel)} physical records, cannot split two arms")
    return sel[:half] if arm == 0 else sel[half:]


def report(path: str) -> None:
    """Print the compare record and the per-arm registration readout for one rig file."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cmps = [r for r in rows if r.get("record") == "compare"]
    if not cmps:
        print(f"{path}: no compare record")
        return
    c = cmps[0]
    print(f"\n=== {pathlib.Path(path).name}")
    print(f"pose_noise {c['pose_noise_cfg']}  keep={c['keep']}  "
          f"REG_N cand={c['candidate_knobs'].get('REG_N')} base={c['baseline_knobs'].get('REG_N')}")
    for s in ("fixture_A", "fixture_B", "fixture_R"):
        d = c.get(s, {})
        print(f"  {s:10s} covc {d.get('mean_a', float('nan')):.4f} -> {d.get('mean_b', float('nan')):.4f} "
              f"(d {d.get('delta', float('nan')):+.4f})  succ {d.get('succ_a')}/{d.get('n_a')} -> "
              f"{d.get('succ_b')}/{d.get('n_b')}  welch {d.get('welch_p')}  fisher {d.get('fisher_p')}")
    for arm, tag in ((0, "cand"), (1, "base")):
        sel = [r for suite in ("fixture_A", "fixture_B", "fixture_R")
               for r in arm_rows(rows, arm, suite)]
        e = [float(r["reg_err_xy_m"]) * 1000.0 for r in sel if r.get("reg_ok")]
        cov = [float(r["coverage_cont"]) for r in sel]
        w = [float(r["wall_s"]) for r in sel]
        marg = [float(r["reg_cn_margin"]) for r in sel if r.get("reg_cn_margin") is not None]
        print(f"  {tag:4s} n_ep {len(sel):3d} reg_ok {sum(1 for r in sel if r.get('reg_ok')):3d} "
              f"reg_err_xy med {st.median(e):6.2f} p90 {pct(e, 0.9):6.2f} mm | "
              f"covc mean {st.mean(cov):.4f} min {min(cov):.4f} | wall/ep {st.mean(w):.3f} s | "
              f"margin med {st.median(marg) if marg else float('nan'):.4f} m")


def probe(path: str) -> None:
    """Print the probe's parity law: error vs n at each sigma."""
    d = json.loads(pathlib.Path(path).read_text())
    print(f"\n=== {pathlib.Path(path).name} (no physics, no coverage, no success)")
    print(f"{'n':>5} {'sigma':>6} {'k':>4} {'rays':>9} {'pitch_mm':>9} {'amp_mm':>7} "
          f"{'reg_ok':>7} {'med_mm':>7} {'p90_mm':>7} {'p90/amp':>8}")
    for r in d["rows"]:
        print(f"{r['n']:5d} {r['sigma_t_m']:6.2f} {r['k']:4d} {r['rays']:9d} {r['ray_pitch_mm']:9.2f} "
              f"{r['parity_amp_mm']:7.2f} {r['reg_ok_rate']:7.3f} {r['err_med_mm']:7.2f} "
              f"{r['err_p90_mm']:7.2f} {r['err_over_parity_p90']:8.2f}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        (probe if a.endswith(".json") else report)(a)
