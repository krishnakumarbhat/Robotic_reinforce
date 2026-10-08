"""I12 (run 246) read-out: 200-seed robustness matrix of the final stack.

Purpose: turn the two 200-seed rig cells (pose_noise 0,0 and 0.01,2; 3 suites x 2 paired
arms) into the paper Table 1 row set: transfer_success, P10 coverage, force compliance,
cycle time -- and re-derive the I22 effect at 10x the seed count with exact CIs.
Inputs: the two rig JSONL files. Outputs: stdout table + a JSON blob appended to stdout.
No physics here: every number is read from the rig's own per-episode records.
"""

from __future__ import annotations

import json
import math
import statistics
import sys


def load(path: str) -> list[dict]:
    """Purpose: read a rig JSONL into records. Inputs: path. Outputs: list[dict]."""
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def episodes(recs: list[dict], n_arms: int = 2) -> dict[tuple[str, int, str], list[dict]]:
    """Purpose: index episode rows by (arm, suite). The rig stamps every episode with
    `path_mode = args.path` for the candidate and `args.compare` for the baseline, which are
    the SAME string when only --compare-env differs, so the arm must come from write order.
    The seed is arm-independent (`BASE_SEED + 1000*(si*seeds) + k`) and the loop is
    arm-major, so the n_arms-th occurrence of a (suite, seed) key is arm n_arms-1.
    Inputs: records, arm count. Outputs: {(arm, suite): [episode, ...]}."""
    seen: dict[tuple[str, int], int] = {}
    out: dict[tuple[str, int, str], list[dict]] = {}
    for r in recs:
        if r.get("record") != "episode":
            continue
        key = (r["suite"], r["seed"])
        arm = seen.get(key, 0)
        seen[key] = arm + 1
        out.setdefault((arm, r["suite"]), []).append(r)
    return out


def q(xs: list[float], p: float) -> float:
    """Purpose: linear-interpolated quantile, numpy 'linear' method. Inputs: xs, p in [0,1]."""
    s = sorted(xs)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    h = (len(s) - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def ci95(xs: list[float]) -> tuple[float, float, float]:
    """Purpose: mean and normal-approx 95% CI half-width. Inputs: xs. Outputs: (mean, half, sd)."""
    n = len(xs)
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n), sd


def binom_ci95(k: int, n: int) -> tuple[float, float]:
    """Purpose: Wilson 95% interval for a success rate (the normal approx is degenerate at
    k=n, which is exactly the regime the champion sits in). Inputs: k, n. Outputs: (lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    """Purpose: build the Table-1 read-out for both noise cells. Inputs: argv paths."""
    result: dict = {}
    for path in sys.argv[1:]:
        recs = load(path)
        hdr = next(r for r in recs if r.get("record") == "header")
        cmp_ = next(r for r in recs if r.get("record") == "compare")
        idx = episodes(recs)
        result["arm_sizes"] = {str(k): len(v) for k, v in sorted(idx.items())}
        cell: dict = {
            "pose_noise_cfg": hdr["pose_noise_cfg"],
            "seeds": hdr["seeds_requested"],
            "episodes": sum(len(v) for v in idx.values()),
            "rig_keep": cmp_.get("keep"),
            "metric": cmp_.get("metric"),
            "cand_label": cmp_["candidate"],
            "base_label": cmp_["baseline"],
            "cand_knobs": cmp_["candidate_knobs"],
            "base_knobs": cmp_["baseline_knobs"],
            "suites": {},
        }
        print(f"\n=== pose_noise {hdr['pose_noise_cfg']} | seeds {hdr['seeds_requested']} "
              f"| keep={cmp_.get('keep')} | metric={cmp_.get('metric')} ===")
        for suite in ("fixture_A", "fixture_B", "fixture_R"):
            ce, be = idx[(0, suite)], idx[(1, suite)]
            cc = [e["coverage_cont"] for e in ce]
            bc = [e["coverage_cont"] for e in be]
            cm, ch, csd = ci95(cc)
            bm, bh, bsd = ci95(bc)
            ck = sum(1 for e in ce if e["success"])
            bk = sum(1 for e in be if e["success"])
            clo, chi = binom_ci95(ck, len(ce))
            blo, bhi = binom_ci95(bk, len(be))
            fr = [e.get("force_compliance", float("nan")) for e in ce]
            fnm = [e.get("fn_mean", float("nan")) for e in ce]
            wall = [e.get("wall_s", float("nan")) for e in ce]
            plen = [e.get("path_len_m", float("nan")) for e in ce]
            st = cmp_[suite]
            cell["suites"][suite] = {
                "n": len(ce), "n_base": len(be),
                "success_cand": ck, "success_base": bk,
                "succ_cand": ck / len(ce), "succ_cand_ci95": [clo, chi],
                "succ_base": bk / len(be), "succ_base_ci95": [blo, bhi],
                "fisher_p_success": st.get("fisher_p_success"),
                "covc_cand": cm, "covc_cand_ci95": [cm - ch, cm + ch], "covc_cand_sd": csd,
                "covc_base": bm, "covc_base_ci95": [bm - bh, bm + bh], "covc_base_sd": bsd,
                "delta_covc": cm - bm,
                "covc_p10": q(cc, 0.10), "covc_p50": q(cc, 0.50), "covc_p90": q(cc, 0.90),
                "covc_min": min(cc), "covc_max": max(cc),
                "base_covc_p10": q(bc, 0.10), "base_covc_min": min(bc),
                "n_below_0.90": sum(1 for x in cc if x < 0.90),
                "welch_p_coverage": st.get("welch_p"),
                "force_compliance_mean": statistics.fmean(fr),
                "force_compliance_p10": q(fr, 0.10),
                "force_compliance_frac_ge_0.90": sum(1 for x in fr if x >= 0.90) / len(fr),
                "fn_mean": statistics.fmean(fnm), "fn_p95_mean": statistics.fmean(
                    [e.get("fn_p95", float("nan")) for e in ce]),
                "cycle_wall_s_mean": statistics.fmean(wall),
                "cycle_wall_s_p90": q(wall, 0.90),
                "path_len_m_mean": statistics.fmean(plen),
                "slip_m_mean": statistics.fmean([e["slip_m"] for e in ce]),
                "escaped_frac": sum(1 for e in ce if e.get("escaped")) / len(ce),
                "stall_frac_mean": statistics.fmean([e["stall_frac"] for e in ce]),
                "jerk_mean": statistics.fmean([e["jerk"] for e in ce]),
                "friction_min": min(e["friction"] for e in ce),
                "friction_max": max(e["friction"] for e in ce),
                "tool_ids": sorted({e["tool_id"] for e in ce}),
            }
            s = cell["suites"][suite]
            print(f"{suite}: succ {bk}/{len(be)} -> {ck}/{len(ce)} "
                  f"({bk / len(be):.3f} -> {ck / len(ce):.3f}) Wilson95 cand "
                  f"[{clo:.4f},{chi:.4f}]  Fisher p={st.get('fisher_p_success')}")
            print(f"    covc  {bm:.4f} -> {cm:.4f}  d={s['delta_covc']:+.4f}  "
                  f"Welch p={st.get('welch_p'):.3e}  cand p10={s['covc_p10']:.4f} "
                  f"min={s['covc_min']:.4f} sd={csd:.4f}  n<0.90: {s['n_below_0.90']}/{len(ce)}"
                  f"  base p10={s['base_covc_p10']:.4f} min={s['base_covc_min']:.4f}")
            print(f"    force_compliance {s['force_compliance_mean']:.4f} "
                  f"(p10 {s['force_compliance_p10']:.4f}, frac>=0.90 "
                  f"{s['force_compliance_frac_ge_0.90']:.3f})  fn_mean {s['fn_mean']:.4f} N  "
                  f"fn_p95 {s['fn_p95_mean']:.4f} N")
            print(f"    cycle {s['cycle_wall_s_mean']:.4f} s (p90 {s['cycle_wall_s_p90']:.4f})  "
                  f"path {s['path_len_m_mean']:.4f} m  slip {s['slip_m_mean']:.4f} m  "
                  f"jerk {s['jerk_mean']:.5f}  mu [{s['friction_min']},{s['friction_max']}]  "
                  f"tools {s['tool_ids']}")
        result[hdr["pose_noise_cfg"]] = cell
    print("\n=== I12_JSON ===")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
