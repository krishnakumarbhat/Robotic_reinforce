"""I12 (run 246) companion: what does the 200-seed force-compliance column actually say?

Purpose: `force_compliance` is the fraction of scrub ticks with normal force inside
[0.5, 1.5] x FN_SET_N. The 20-seed runs reported a MEAN (0.4787 on fixture_A) and the
strategy graph named it an open gap. A mean over a bimodal distribution is not a
diagnosis. This stratifies the same 200-seed episodes the matrix already wrote -- no new
run, no physics -- by tool_id, friction decade and success, to say whether the gap is a
tail (a few bad seeds) or the bulk (a structural offset).
Inputs: the two rig JSONL files. Outputs: stdout table.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def arm_split(recs: list[dict], n_arms: int = 2) -> list[dict]:
    """Purpose: return candidate-arm episodes in write order. The rig stamps both arms with
    the same `path_mode` when only --compare-env differs; the seed formula is arm-independent
    and the loop is arm-major, so the first occurrence of each (suite, seed) is the candidate.
    Inputs: records. Outputs: list of candidate-arm episode dicts."""
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for r in recs:
        if r.get("record") != "episode":
            continue
        key = (r["suite"], r["seed"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main() -> int:
    """Purpose: stratify force_compliance. Inputs: argv paths."""
    for path in sys.argv[1:]:
        recs = [json.loads(l) for l in open(path) if l.strip()]
        hdr = next(r for r in recs if r.get("record") == "header")
        eps = arm_split(recs)
        print(f"\n=== pose_noise {hdr['pose_noise_cfg']} | candidate arm only "
              f"({len(eps)} episodes) | fn band [0.25, 0.75] N (FN_SET_N="
              f"{hdr.get('fn_set_n', 0.5)}) ===")
        for suite in ("fixture_A", "fixture_B", "fixture_R"):
            se = [e for e in eps if e["suite"] == suite]
            fc = [e["force_compliance"] for e in se]
            print(f"\n{suite}: n={len(se)} fc mean {statistics.fmean(fc):.4f} "
                  f"min {min(fc):.4f} max {max(fc):.4f} "
                  f"frac>=0.90 {sum(1 for x in fc if x >= 0.90) / len(fc):.3f}")
            by_tool: dict[int, list[float]] = defaultdict(list)
            for e in se:
                by_tool[e["tool_id"]].append(e["force_compliance"])
            for t in sorted(by_tool):
                v = by_tool[t]
                print(f"    tool {t}: n={len(v):3d} fc {statistics.fmean(v):.4f} "
                      f"p10 {sorted(v)[max(0, len(v) // 10)]:.4f} "
                      f"frac>=0.90 {sum(1 for x in v if x >= 0.90) / len(v):.3f}")
            by_mu: dict[str, list[float]] = defaultdict(list)
            for e in se:
                by_mu[f"{int(e['friction'] * 10) / 10:.1f}"].append(e["force_compliance"])
            line = "    mu decade fc: " + "  ".join(
                f"{k}:{statistics.fmean(v):.3f}(n{len(v)})" for k, v in sorted(by_mu.items()))
            print(line)
            # is the distribution bimodal? gap test on the sorted fc values
            sv = sorted(fc)
            gaps = [(sv[i + 1] - sv[i], i) for i in range(len(sv) - 1)]
            gi, _ = max(gaps)
            print(f"    largest gap {gi:.4f} after rank {gi and _}/{len(sv)} "
                  f"-> {'BIMODAL' if gi > 0.2 else 'unimodal'}; "
                  f"fc<0.50: {sum(1 for x in fc if x < 0.50) / len(fc):.3f}  "
                  f"fc>=0.90: {sum(1 for x in fc if x >= 0.90) / len(fc):.3f}")
            fnm = [e["fn_mean"] for e in se]
            fnp = [e["fn_p95"] for e in se]
            print(f"    fn_mean {statistics.fmean(fnm):.4f} (p10 {sorted(fnm)[len(fnm) // 10]:.4f} "
                  f"max {max(fnm):.4f})  fn_p95 {statistics.fmean(fnp):.4f} "
                  f"(p10 {sorted(fnp)[len(fnp) // 10]:.4f} max {max(fnp):.4f})")
            # correlation fc vs fn_mean / fn_p95: if the head simply presses harder,
            # fc is set by the band width, not by a control failure
            def pearson(xs: list[float], ys: list[float]) -> float:
                n = len(xs)
                mx, my = statistics.fmean(xs), statistics.fmean(ys)
                sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                sxx = sum((x - mx) ** 2 for x in xs)
                syy = sum((y - my) ** 2 for y in ys)
                return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")
            print(f"    corr(fc, fn_mean) {pearson(fc, fnm):+.3f}  "
                  f"corr(fc, fn_p95) {pearson(fc, fnp):+.3f}  "
                  f"corr(fc, success) n/a (binary)")
            ok = [e["force_compliance"] for e in se if e["success"]]
            bad = [e["force_compliance"] for e in se if not e["success"]]
            fmean = lambda xs: statistics.fmean(xs) if xs else float("nan")  # noqa: E731
            print(f"    fc | success {fmean(ok):.4f} (n{len(ok)})  "
                  f"fc | fail {fmean(bad):.4f} (n{len(bad)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
