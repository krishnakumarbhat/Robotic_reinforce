"""N192 readout: paired multi-cast (AEGIS_REG_CASTS=0) vs frozen single cast (=1) on the
same seeds. Reads only the rig's own summary/compare records; prints no derived metric that
the rig did not compute (G7).

Purpose: dump the per-suite paired compare record + the registration-error distribution of
both arms for every noise level of run 303.
Inputs: results/aegis_v2/N192_r303_s100_*.jsonl. Outputs: a table (stdout).
"""
import glob
import json
import statistics as st

for path in sorted(glob.glob("results/aegis_v2/N192_r303_s100_*.jsonl")):
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    hdr = [r for r in rows if r.get("record") == "header"][0]
    comp = [r for r in rows if r.get("record") == "compare"][0]
    eps = [r for r in rows if r.get("record") == "episode"]
    print(f"\n=== {path.split('/')[-1]}  pose_noise={hdr['pose_noise_cfg']}  "
          f"reg_casts={hdr.get('aegis_reg_casts')} H={hdr['aegis_reg_half_m']} "
          f"{len(eps)//2} episodes/arm")
    print(f"{'suite':10s} {'covc a->b':>18s} {'succ a->b':>12s} {'welch_p':>10s} "
          f"{'fisher_p':>10s} {'paired_p':>10s}")
    for s in ("fixture_A", "fixture_B", "fixture_R"):
        c = comp[s]
        print(f"{s:10s} {c['mean_a']:.4f}->{c['mean_b']:.4f} d={c['delta']:+.4f}  "
              f"{c['succ_a']:3d}->{c['succ_b']:3d}  {c['welch_p']:.3e}  {c['fisher_p']:.3e}  "
              f"{c['paired_p']:.3e}")
    print("  rig keep:", comp.get("keep"), "| keep_rule:", comp.get("keep_rule"))
    for arm, sub in (("dose", eps[: len(eps) // 2]), ("frozen", eps[len(eps) // 2:])):
        for s in ("fixture_B", "fixture_R"):
            e = [r for r in sub if r["suite"] == s and r.get("reg_err_xy_m") == r.get(
                "reg_err_xy_m")]
            err = sorted(r["reg_err_xy_m"] for r in e)
            ok = sum(1 for r in e if r["reg_ok"])
            kc = sorted({r["reg_casts"] for r in e})
            win = sum(1 for r in e if max(abs(x) for x in r["reg_cast_win_m"]) > 1e-9)
            pts = [r["reg_cast_pts"] for r in e]
            print(f"  {arm:6s} {s:10s} k={kc} reg_err mm med {1000*st.median(err):6.2f} "
                  f"p90 {1000*err[int(0.9*len(err))-1]:6.2f} max {1000*err[-1]:7.2f} | "
                  f"reg_ok {ok}/{len(e)} | off-centre winner {win}/{len(e)} | "
                  f"win pts med {st.median(pts):.0f}")
    print("  harness errors:", sum(1 for r in eps if "HARNESS" in str(r.get("status", ""))))
