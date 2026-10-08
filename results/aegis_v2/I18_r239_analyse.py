"""I18 paired analysis: summarise BOTH arms of a `--compare` rig JSONL for the log.

Purpose: one place to print per-arm per-suite transfer_success / coverage_cont / slip /
    path_len for the candidate and the paired champion, and to echo the rig's OWN compare
    record (Welch p on coverage_cont, Fisher p on success, keep flag). It reads records and
    copies p-values -- it never recomputes or rescales a metric (G7).
Inputs: results/aegis_v2/<file>.jsonl. Outputs: <stem>_result.json + a printed table.
"""
import json
import os
import sys

NUM = ("coverage_cont", "slip_m", "force_compliance", "fn_mean", "path_len_m", "max_turn_deg", "steps")


def main():
    path = sys.argv[1]
    hdr = rec = None
    eps = []
    for line in open(path):
        d = json.loads(line)
        if d.get("record") == "header":
            hdr = d
        elif d.get("record") == "compare":
            rec = d
        elif d.get("record") == "episode":
            eps.append(d)
    arms = {}
    for e in eps:
        arms.setdefault(e["path_mode"], []).append(e)
    out = {"file": os.path.basename(path), "pose_noise_cfg": hdr["pose_noise_cfg"],
           "path_mode": hdr["path_mode"], "compare": hdr["compare"],
           "seeds": hdr["seeds_requested"], "suites": hdr["suites"], "rig_version": hdr["rig_version"],
           "arms": {}}
    print(f"== {os.path.basename(path)} pose_noise={hdr['pose_noise_cfg']} path={hdr['path_mode']} "
          f"compare={hdr['compare']} seeds={hdr['seeds_requested']} suites={hdr['suites']}")
    for arm, rows in arms.items():
        per = {}
        for s in hdr["suites"]:
            sel = [r for r in rows if r["suite"] == s]
            if not sel:
                continue
            cov = [float(r["coverage_cont"]) for r in sel]
            per[s] = {"n": len(sel), "success_k": f"{sum(bool(r['success']) for r in sel)}/{len(sel)}",
                      "transfer_success": round(sum(bool(r["success"]) for r in sel) / len(sel), 4),
                      "coverage_cont": round(sum(cov) / len(cov), 4),
                      "coverage_min": round(min(cov), 4),
                      "n_eps_at_ceiling": sum(1 for c in cov if c >= 0.9999),
                      "slip_m": round(sum(float(r["slip_m"]) for r in sel) / len(sel), 5),
                      "slip_max": round(max(float(r["slip_m"]) for r in sel), 5),
                      "path_len_m": sel[0].get("path_len_m"),
                      "force_compliance": round(sum(float(r.get("force_compliance", 0.0)) for r in sel) / len(sel), 4),
                      "escaped": sum(1 for r in sel if r.get("escaped"))}
        out["arms"][arm] = {"episodes": len(rows), "per_suite": per,
                            "statuses": sorted({r.get("status") for r in rows}),
                            "pts_sources": sorted({r.get("pts_source") for r in rows}),
                            "max_turn_deg": max([float(r.get("max_turn_deg") or 0.0) for r in rows] or [0.0]),
                            "steps_mean": round(sum(int(r.get("steps", 0)) for r in rows) / len(rows), 1)}
        print(f"-- arm {arm}: {len(rows)} eps, statuses {out['arms'][arm]['statuses']}, "
              f"pts_source {out['arms'][arm]['pts_sources']}, max_turn {out['arms'][arm]['max_turn_deg']:.2f} deg, "
              f"steps mean {out['arms'][arm]['steps_mean']}")
        for s, v in per.items():
            print(f"   {s:10s} succ {v['success_k']:>6s} ({v['transfer_success']:.4f})  "
                  f"covc {v['coverage_cont']:.4f} (min {v['coverage_min']:.4f}, {v['n_eps_at_ceiling']} at ceiling)  "
                  f"slip {v['slip_m']:.5f} (max {v['slip_max']:.5f})  len {v['path_len_m']}  "
                  f"fnc {v['force_compliance']:.3f}  esc {v['escaped']}")
    if rec:
        out["rig_compare_record"] = rec
        print(f"-- rig compare record: metric={rec['metric']} keep={rec['keep']}")
        print(f"   rule: {rec['keep_rule']}")
        for s in hdr["suites"]:
            if s in rec:
                c = rec[s]
                # the rig's mean_a is the BASELINE (champion) arm and mean_b the CANDIDATE arm
                print(f"   {s:10s} champ {c['mean_a']:.4f} -> cand {c['mean_b']:.4f} "
                      f"delta {c['delta']:+.4f} Welch p {c['welch_p']:.3g} paired p {c['paired_p']:.3g} | "
                      f"succ {c['succ_a']}/{c['n_a']} -> {c['succ_b']}/{c['n_b']} Fisher p {c['fisher_p']:.3g}")
    dst = path.replace(".jsonl", "_result.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
