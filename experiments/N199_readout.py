"""N199 readout: held-out seed-base replication of the N197 saturation + grid quantisation floor.

Purpose: adjudicate whether fixture_B's saturation is seed-base-independent, and whether the
primary metric coverage_cont has enough resolution left to discriminate anything.
Inputs:  N199_r311_s100_*.jsonl (held-out AEGIS_BASE_SEED=250000), N197_r310_s100_*.jsonl (base 91000).
Outputs: one JSON blob on stdout with per-level per-suite success / coverage_cont, the exact
         coverage_cont denominator, and the N197-vs-N199 paired deltas. No physics, no scoring.
"""
import glob, json, os, sys
from collections import Counter


def load(path):
    lines = open(path).read().strip().split("\n")
    hdr = json.loads(lines[0])
    eps = [json.loads(l) for l in lines[1:] if json.loads(l).get("record") == "episode"]
    summ = json.loads(lines[-1])
    return hdr, eps, summ


def tag_arms(eps):
    """Split the two paired arms of one sweep.

    Rig semantics: the process env is the CANDIDATE arm and --compare-env the BASELINE arm. Here the
    process env carried REG_CASTS=0 (frozen single cast) and the compare arm REG_CASTS=1 (N192 lattice).
    The rig emits the baseline episode first for each (suite, seed), so the ordinal is the reliable key:
    reg_casts alone cannot separate the arms below the k-law breakpoint (k=ceil(6 sigma/d)+1 = 1 at 0.03 m).
    """
    seen, out = {}, []
    for e in eps:
        k = (e["suite"], e["seed"])
        i = seen.get(k, 0)
        seen[k] = i + 1
        e = dict(e)
        e["arm"] = "lattice" if i % 2 == 0 else "single_cast"
        out.append(e)
    return out


def quantiser(vals):
    """Smallest integer D with every coverage_cont on the grid k/D (tol 1e-6)."""
    for D in range(2, 4097):
        if all(abs(v * D - round(v * D)) < 1e-6 for v in vals):
            return D
    return None


def level_stats(eps, suites=("fixture_A", "fixture_B", "fixture_R")):
    eps = tag_arms(eps)
    out = {}
    for s in suites:
        for arm in ("single_cast", "lattice"):
            e = [x for x in eps if x["suite"] == s and x["arm"] == arm]
            if not e:
                continue
            cov = [x["coverage_cont"] for x in e]
            out[f"{s}/{arm}"] = {
                "n": len(e),
                "success": round(sum(x["success"] for x in e) / len(e), 4),
                "covc": round(sum(cov) / len(cov), 4),
                "covc_min": round(min(cov), 4),
                "covc_p10": round(sorted(cov)[max(0, int(0.1 * len(cov)) - 1)], 4),
                "covc_k_over_128": sorted({round(v * 128) for v in cov})[:4],
                "reg_err_xy_med": round(sorted(x["reg_err_xy_m"] for x in e)[len(e) // 2], 5),
                "casts": sorted({int(x.get("reg_casts") or 0) for x in e}),
                "reg_ok": round(sum(bool(x["reg_ok"]) for x in e) / len(e), 4),
            }
    return out


def main():
    res = {}
    for tag, lvl in (("0.03,6", "0.03,6"), ("0.28,56", "0.28,56"), ("3.2,640", "3.2,640")):
        n199 = "results/aegis_v2/N199_r311_s100_" + tag.replace(",", "_") + ".jsonl"
        n197 = "results/aegis_v2/N197_r310_s100_" + tag + ".jsonl"
        if not os.path.exists(n199):
            continue
        h9, e9, s9 = load(n199)
        h7, e7, s7 = load(n197)
        assert h9["pose_noise_cfg"] == h7["pose_noise_cfg"] == lvl, (h9["pose_noise_cfg"], h7["pose_noise_cfg"])
        # rig identity: every non-timing header field must match between the two runs
        skip = {"ts"}
        drift = {k: [h7[k], h9[k]] for k in h7 if k not in skip and h7[k] != h9.get(k)}
        lvl_res = {
            "level": lvl,
            "base_seed": {"N197_certs": 91000, "N199_heldout": 250000},
            "rig_header_drift_vs_N197": drift,
            "N199": level_stats(e9),
            "N197": level_stats(e7),
            "compare_N199": s9.get("compare") or next((json.loads(x) for x in open(n199).read().strip().split("\n")[1:-1] if json.loads(x).get("record") == "compare"), None),
            "compare_N197": next((json.loads(x) for x in open(n197).read().strip().split("\n")[1:-1] if json.loads(x).get("record") == "compare"), None),
        }
        for s in ("fixture_A", "fixture_B", "fixture_R"):
            a = lvl_res["N199"].get(f"{s}/lattice")
            b = lvl_res["N197"].get(f"{s}/lattice")
            if a and b:
                lvl_res.setdefault("delta_heldout_minus_cert_lattice", {})[s] = {
                    "d_success": round(a["success"] - b["success"], 4),
                    "d_covc": round(a["covc"] - b["covc"], 4),
                }
        lvl_res["coverage_cont_grid_D_per_suite"] = {
            s: quantiser([x["coverage_cont"] for x in tag_arms(e9 + e7)
                          if x["suite"] == s and x["arm"] == "lattice"])
            for s in ("fixture_A", "fixture_B", "fixture_R")
        }
        res[lvl] = lvl_res
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
