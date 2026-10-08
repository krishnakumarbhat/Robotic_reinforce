"""N479 readout: per-cell paired candidate vs open-loop baseline, force-channel statistics."""
import json
import math
import statistics as st
import sys


def split(path):
    rows = [json.loads(l) for l in open(path)]
    eps = [r for r in rows if r.get("record") == "episode"]
    modes = [r for r in rows if r.get("record") == "summary_mode"]
    n = sum(s["n"] for s in modes[0]["per_suite"].values())
    assert n * 2 == len(eps), (n, len(eps))
    return eps[:n], eps[n:], next((r for r in rows if r.get("record") == "compare")), rows


def blk(arm, suite):
    s = [r for r in arm if r["suite"] == suite]
    return dict(
        n=len(s), succ=sum(r["success"] for r in s),
        covc=st.mean(r["coverage_cont"] for r in s),
        fn=st.mean(r["fn_mean"] for r in s), fnstd=st.mean(r["fn_std"] for r in s),
        fnp95=st.mean(r["fn_p95"] for r in s), compl=st.mean(r["force_compliance"] for r in s),
        press=st.mean(r["press_mean_n"] for r in s), pmax=max(r["press_max_n"] for r in s),
        slip=st.mean(r["slip_m"] for r in s), stall=st.mean(r["stall_frac"] for r in s),
        clamp=st.mean(r["f_clamp_ticks"] for r in s), esc=sum(r["escaped"] for r in s),
        wall=st.mean(r["wall_s"] for r in s),
        harn=sum(1 for r in s if r.get("backend") != "pybullet"))


def main(paths):
    for f in paths:
        c, b, cmp, rows = split(f)
        h = rows[0]
        nm = f.split("/")[-1].replace("N479_r497_", "").replace(".jsonl", "")
        print(f"\n=== {nm}  pose={h.get('pose_noise_cfg')}  "
              f"PI={h.get('force_pi')} set={h.get('fn_set_n')} kp={h.get('fn_kp')} ki={h.get('fn_ki')}")
        print(f"{'suite':10s} {'cSucc':>7s} {'bSucc':>7s} {'cCovc':>7s} {'bCovc':>7s} "
              f"{'cFn':>6s} {'bFn':>6s} {'fnStdRatio':>10s} {'cCompl':>7s} {'bCompl':>7s} "
              f"{'cPress':>7s} {'cPmax':>6s} {'cSlip':>8s} {'bSlip':>8s} {'esc':>4s} {'harn':>4s}")
        for suite in ("fixture_A", "fixture_B", "fixture_R"):
            x, y = blk(c, suite), blk(b, suite)
            ratio = x["fnstd"] / y["fnstd"] if y["fnstd"] else float("nan")
            print(f"{suite:10s} {x['succ']:3d}/{x['n']:<3d} {y['succ']:3d}/{y['n']:<3d} "
                  f"{x['covc']:7.4f} {y['covc']:7.4f} {x['fn']:6.3f} {y['fn']:6.3f} {ratio:10.3f} "
                  f"{x['compl']:7.4f} {y['compl']:7.4f} {x['press']:7.3f} {x['pmax']:6.3f} "
                  f"{x['slip']:8.5f} {y['slip']:8.5f} {x['esc']:4d} {x['harn'] + y['harn']:4d}")
        for suite in ("fixture_A", "fixture_B", "fixture_R"):
            cc = (cmp or {}).get(suite, {})
            if cc:
                print(f"  cmp {suite:10s} welch_p={cc.get('p_value')} fisher_p={cc.get('fisher_p')} "
                      f"succ_c={cc.get('cand_success')} succ_b={cc.get('base_success')} "
                      f"covc_c={cc.get('cand_mean')} covc_b={cc.get('base_mean')}")
        print(f"  keep={None if not cmp else cmp.get('keep')}")


if __name__ == "__main__":
    main(sys.argv[1:])
