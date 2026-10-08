"""Purpose: read the I3 probe/headroom JSONL arms and print the paired per-suite table.
Inputs: paths. Outputs: stdout table only (no file writes). ponytail: stdlib + statistics."""
import json
import statistics
import sys


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return rows[0], [r for r in rows if r.get("record") == "episode"]


def arm(path, label):
    h, eps = load(path)
    out = {}
    for s in ("fixture_A", "fixture_B", "fixture_R"):
        e = [r for r in eps if r["suite"] == s]
        cov = [r["coverage_cont"] for r in e]
        out[s] = {
            "n": len(e), "succ": f"{sum(r['success'] for r in e)}/{len(e)}",
            "covc": round(statistics.fmean(cov), 4), "min": round(min(cov), 4),
            "sd": round(statistics.stdev(cov), 4) if len(cov) > 1 else 0.0,
            "rms_mm": round(1000 * statistics.fmean([r.get("residual_rms_m", 0) for r in e]), 2),
            "kernel_gap": round(statistics.fmean([r.get("cov_kernel_gap", 0) for r in e]), 4),
            "slip": round(statistics.fmean([r["slip_m"] for r in e]), 4),
            "harness": sum(1 for r in e if not str(r.get("status", "")).startswith("PHYSICAL")),
        }
    return out, h


if __name__ == "__main__":
    labels, table = [], {}
    for i in range(1, len(sys.argv), 2):
        p, lab = sys.argv[i], sys.argv[i + 1]
        table[lab], _ = arm(p, lab)
        labels.append(lab)
    keys = ("n", "succ", "covc", "min", "sd", "rms_mm", "kernel_gap", "slip", "harness")
    print(f"{'suite':<10}{'arm':<14}" + "".join(f"{k:>12}" for k in keys))
    for s in ("fixture_A", "fixture_B", "fixture_R"):
        for lab in labels:
            print(f"{s:<10}{lab:<14}" + "".join(f"{str(table[lab][s][k]):>12}" for k in keys))
