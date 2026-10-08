"""Read N681 rig JSONL cells and print the rig's OWN decision fields.

Purpose: one reader for every cell of the N681 registration-window ladder so that
no number in the worklog, equations.md or the JSONL result row is hand-transcribed.
Success and coverage are reported exactly as the rig computed them from physics
contact points (`pts_source`); this script never rescales, re-averages across arms
or estimates them. Per-arm aggregates come from the rig's own `summary_mode`
records, never from a re-aggregation here.

Usage: python3 experiments/N681_read.py results/aegis_v2/N681_*.jsonl [...]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict


def _f(x: object) -> float:
    """Coerce to float, mapping None/'' to nan."""
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _median(vals: list[float]) -> float:
    """Median of a finite list, nan if empty."""
    v = sorted(x for x in vals if not math.isnan(x))
    return v[len(v) // 2] if v else float("nan")


def summarise(path: str) -> dict:
    """Return header, per-arm rig aggregates, cast/estimator readouts and compare."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    header = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    modes = [r for r in rows if r.get("record") == "summary_mode"]
    cmp_row = next((r for r in rows if r.get("record") == "compare"), None)

    # A candidate cell derives k from the pose-noise label (reg_casts = k^2 >> 1);
    # the frozen single-cast baseline arm logs reg_casts == 1.
    cand = [r for r in eps if (_f(r.get("reg_casts")) or 0.0) > 1.5]
    base = [r for r in eps if (_f(r.get("reg_casts")) or 0.0) <= 1.5]

    out = {"path": path, "header": header, "episodes": len(eps),
           "harness_errors": sum(1 for r in eps
                                 if str(r.get("status", "")).startswith("HARNESS")),
           "pts_source": sorted({str(r.get("pts_source")) for r in eps}),
           "cand": _arm(cand), "base": _arm(base), "modes": len(modes)}

    # rig-side per-arm aggregates (authoritative; used for any reported mean)
    out["arm_summaries"] = []
    for m in modes:
        ps = m.get("per_suite", {})
        out["arm_summaries"].append({
            "path_mode": m.get("run", {}).get("path_mode"),
            "episodes": m.get("run", {}).get("episodes"),
            "per_suite": {k: {"n": v.get("n"), "succ": v.get("success_count"),
                              "succ_rate": v.get("transfer_success"),
                              "cov": v.get("mean_coverage_cont"),
                              "cov_std": v.get("std_coverage_cont"),
                              "err_p50_mm": v.get("reg_err_xy_p50_mm"),
                              "err_p90_mm": v.get("reg_err_xy_p90_mm"),
                              "escaped": v.get("escaped_frac")}
                          for k, v in ps.items()},
        })
    if cmp_row is not None:
        out["compare"] = {k: cmp_row.get(k) for k in ("candidate", "baseline", "keep", "keep_rule")}
        out["compare_suites"] = {k: cmp_row.get(k) for k in ("fixture_A", "fixture_B", "fixture_R")
                                 if k in cmp_row}
    return out


def _arm(rs: list[dict]) -> dict:
    """Per-suite readouts for one arm, from that arm's own episode records."""
    per: dict = defaultdict(list)
    for r in rs:
        per[r.get("suite", "?")].append(r)
    out = {}
    for suite, ss in sorted(per.items()):
        ok = [r for r in ss if not str(r.get("status", "")).startswith("HARNESS")]
        out[suite] = {
            "n": len(ok),
            "success": sum(1 for r in ok if r.get("success")),
            "cov": [_f(r.get("coverage_cont")) for r in ok],
            "reg_ok": sum(1 for r in ok if r.get("reg_ok") is True),
            "casts": sorted({int(_f(r.get("reg_casts"))) for r in ok
                             if not math.isnan(_f(r.get("reg_casts")))}),
            "err_mm": [1000.0 * _f(r.get("reg_err_xy_m")) for r in ok],
            "wall_s": [_f(r.get("wall_s")) for r in ok],
            "tags": sorted({str(r.get("quality_tag")) for r in ok}),
        }
        out[suite]["cov_mean_ep"] = (sum(out[suite]["cov"]) / len(out[suite]["cov"])
                                     if out[suite]["cov"] else float("nan"))
        out[suite]["err_p50_mm"] = _median(out[suite]["err_mm"])
        out[suite]["wall_s_mean"] = (sum(out[suite]["wall_s"]) / len(out[suite]["wall_s"])
                                    if out[suite]["wall_s"] else float("nan"))
    return out


def _p(v: object) -> str:
    """Format a possibly-nan float for a one-line print."""
    f = _f(v)
    return "nan" if math.isnan(f) else f"{f:.4g}"


def main() -> int:
    for path in sys.argv[1:]:
        o = summarise(path)
        h = o["header"]
        print(f"== {path}")
        print(f"   cfg pose_noise={h.get('pose_noise_cfg')} H={h.get('aegis_reg_half_m')} "
              f"dfact={h.get('aegis_reg_dfact')} cn={h.get('aegis_reg_cn')} "
              f"casts_mode={h.get('aegis_reg_casts')} seeds={h.get('seeds_requested')} "
              f"rig_v={h.get('rig_version')} eps={o['episodes']} "
              f"harness={o['harness_errors']} pts={o['pts_source']}")
        for arm in ("cand", "base"):
            for suite, v in o[arm].items():
                print(f"   [{arm}] {suite:9s} n={v['n']:3d} succ={v['success']:3d} "
                      f"cov_ep={_p(v['cov_mean_ep'])} reg_ok={v['reg_ok']:3d} "
                      f"casts={v['casts']} err_p50={_p(v['err_p50_mm'])}mm "
                      f"wall={_p(v['wall_s_mean'])}s tags={','.join(v['tags'])}")
        for m in o["arm_summaries"]:
            ps = m["per_suite"]
            print(f"   RIG[{m['path_mode'][:46]}] eps={m['episodes']} " + " | ".join(
                f"{k}: succ={v['succ']}/{v['n']} cov={_p(v['cov'])} err_p50={_p(v['err_p50_mm'])}"
                for k, v in ps.items()))
        if "compare_suites" in o:
            b = o["compare_suites"].get("fixture_B", {})
            print(f"   COMPARE B: mean_a(base)={_p(b.get('mean_a'))} mean_b(cand)={_p(b.get('mean_b'))} "
                  f"succ_a={b.get('succ_a')} succ_b={b.get('succ_b')} "
                  f"welch_p={_p(b.get('welch_p'))} fisher_p={_p(b.get('fisher_p'))} "
                  f"keep={o['compare'].get('keep')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
