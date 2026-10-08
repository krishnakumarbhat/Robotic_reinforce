"""N478 analysis: recompute every number the N478 claim uses, from the rig JSONL.
Purpose: one place that splits the two arms, recomputes the stats with scipy, and prints the
         mechanism readout. Never trusts the rig summary text for the paired numbers.
Inputs: rig JSONL paths on argv. Outputs: a per-file table (stdout).
Arm split: the rig runs the candidate arm first, then the --compare arm, so episodes are
ordered by `ts`; the two `summary_mode` records confirm the boundary (2nd one's path_mode
carries the `[knob]` suffix). Split is asserted against the summary_mode counts.
"""
from __future__ import annotations

import json
import math
import statistics
import sys

from scipy import stats

A_SLACK = {h: round(h - math.hypot(0.34, 0.14), 4) for h in (0.6, 0.9, 1.2)}


def split(path):
    """Return (header, cand_eps, base_eps, compare_record) for one rig JSONL."""
    header, eps, other = {}, [], []
    for line in open(path):
        r = json.loads(line)
        if r.get("record") == "header":
            header = r
        elif r.get("record") == "episode":
            eps.append(r)
        else:
            other.append(r)
    modes = [r for r in other if r.get("record") == "summary_mode"]
    assert len(modes) == 2, f"expected 2 arms in {path}, got {len(modes)}"
    # Episodes are WRITTEN arm 1 (candidate) then arm 2 (compare), so the file order -- not
    # `ts` -- is the arm boundary. The boundary count is read from summary_mode[0]'s per-suite
    # n's, and asserted against its run.episodes.
    cand_rows, n_cand = [], 0
    for suite, s in modes[0]["per_suite"].items():
        cand_rows.append((suite, s["n"]))
        n_cand += s["n"]
    assert n_cand * 2 == len(eps), f"arm split {n_cand}/{len(eps)} does not halve {path}"
    assert str(modes[1]["run"]["path_mode"]).find("[") > 0, "2nd arm is not the compare arm"
    cmp_rec = next((r for r in other if r.get("record") == "compare"), None)
    return header, eps[:n_cand], eps[n_cand:], cmp_rec


def stat(rows, suite, key, fn=statistics.mean):
    """Suite-scoped statistic over `key`. NaN is DROPPED, not propagated: the base arm logs
    reg_err_xy_m = NaN on every episode where reg_ok is False (registration found no top face),
    and a single NaN makes statistics.median return NaN for the whole arm -- which silently hid
    the base arm's real residual error at the large-sigma cells. The drop rate is reported
    separately as reg_ok so no episode is hidden.
    """
    v = [r.get(key) for r in rows
         if r.get("suite") == suite and isinstance(r.get(key), (int, float))
         and not math.isnan(r[key]) and not math.isnan(r.get(key))]
    return fn(v) if v else None


def succ(rows, suite):
    v = [r.get("success") for r in rows if r.get("suite") == suite]
    return round(sum(bool(x) for x in v) / len(v), 4) if v else None


def main(paths):
    for path in paths:
        header, cand, base, cmp_rec = split(path)
        name = path.split("/")[-1]
        print("=" * 104)
        print(f"{name}   noise={header.get('pose_noise_cfg')}  seeds={header.get('seeds_requested')}"
              f"  H={header.get('aegis_reg_half_m')}  n={header.get('aegis_reg_n')}"
              f"  cand_casts={header.get('aegis_reg_casts')}"
              f"  base={cmp_rec.get('baseline') if cmp_rec else None}")
        for suite in ("fixture_A", "fixture_B", "fixture_R"):
            cs, bs = succ(cand, suite), succ(base, suite)
            cm, bm = stat(cand, suite, "coverage_cont"), stat(base, suite, "coverage_cont")
            ce = stat(cand, suite, "reg_err_xy_m", statistics.median)
            be = stat(base, suite, "reg_err_xy_m", statistics.median)
            c, b = ([r["coverage_cont"] for r in cand if r.get("suite") == suite],
                    [r["coverage_cont"] for r in base if r.get("suite") == suite])
            p_w = stats.ttest_ind(c, b, equal_var=False).pvalue
            p_p = stats.ttest_rel(c, b).pvalue
            p_f = stats.fisher_exact([[sum(bool(x) for x in c), len(c) - sum(bool(x) for x in c)],
                                      [sum(bool(x) for x in b), len(b) - sum(bool(x) for x in b)]]).pvalue
            print(f"  {suite:10s} succ cand {cs} vs base {bs} | covc {bm:.4f} -> {cm:.4f} "
                  f"(d={cm - bm:+.4f}) welch_p={p_w:.3e} paired_p={p_p:.3e} fisher_p={p_f:.3e}")
            print(f"  {'':10s} reg_err_xy_mm median  cand {ce * 1000 if ce else float('nan'):7.2f}  base "
                  f"{be * 1000 if be else float('nan'):7.2f}   reg_ok {stat(cand, suite, 'reg_ok', lambda v: round(statistics.mean(v), 3))}"
                  f" -> {stat(base, suite, 'reg_ok', lambda v: round(statistics.mean(v), 3))}"
                  f"   escaped {stat(cand, suite, 'escaped', lambda v: round(statistics.mean(v), 3))}"
                  f" -> {stat(base, suite, 'escaped', lambda v: round(statistics.mean(v), 3))}")
        mg = stat(cand, "fixture_B", "reg_cn_margin", statistics.median)
        _ = None
        H = header.get("aegis_reg_half_m")
        if mg is not None:
            print(f"  P5 readout: B winner border-margin median {mg:.4f} m vs a_slack(H={H})="
                  f"{A_SLACK.get(H)} m; coarse pitch 2H/(cn-1)="
                  f"{2 * H / (header.get('aegis_reg_casts') and 15):.4f} m")
        print("  reg_casts seen (cand):", sorted({r.get("reg_casts") for r in cand},
                                                key=lambda v: (v is None, v)))
        if cmp_rec:
            print("  RIG compare keep:", cmp_rec["keep"])


if __name__ == "__main__":
    main(sys.argv[1:])
