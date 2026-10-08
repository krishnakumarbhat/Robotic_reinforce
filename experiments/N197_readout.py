"""Purpose: build the N197 CLOSE-OUT CERTIFICATION MATRIX from the six r310 rig JSONLs.
Inputs: results/aegis_v2/N197_r310_s100_<sigma,yaw>.jsonl (candidate = shipped stack
trochoid + N192 lattice H=0.6 + N195 d-law d=2a/CN=16 + N196 n=32; paired baseline =
AEGIS_REG_CASTS=1, the frozen single cast, SAME seeds).
Outputs: prints the per-level/per-suite/per-arm table (transfer_success, coverage_cont,
reg_err_xy median/p90/max, reg_ok rate, casts, wall_s/ep) plus the certified envelope
(the bracketing sigma at which each suite crosses 0.70 per arm), and writes
results/aegis_v2/N197_matrix.json.
"""
import json
import pathlib
import statistics as st

LEVELS = ["0,0", "0.03,6", "0.28,56", "0.64,128", "1.6,320", "3.2,640"]
PROBE = "6.4,1280"  # 20-seed envelope probe only, never a keep row
SUITES = ["fixture_A", "fixture_B", "fixture_R"]
ROOT = pathlib.Path("results/aegis_v2")


def arms(path):
    """Split one rig file into {arm: [episode rows]}. Both arms log path_mode="trochoid",
    so the split is the rig's own ORDER (candidate block, then --compare block), verified
    by the rig's own ORDER (candidate block, then --compare block), verified by
    reg_casts == 1 on 300/300 baseline rows. reg_casts == 1 is NOT a valid candidate-side
    marker: at sigma = 0 the derived lattice is k = 1 by design, so both blocks read 1.
    """
    rows = [json.loads(ln) for ln in path.read_text().splitlines()
            if json.loads(ln).get("record") == "episode"]
    half = len(rows) // 2
    out = {"dose": rows[:half], "frozen": rows[half:]}
    assert all(r.get("reg_casts", 0) == 1 for r in out["frozen"]), "baseline block not reg_casts==1"
    return out


def stats(rows):
    """Purpose: one arm x suite summary. Returns None if the suite has no episodes."""
    if not rows:
        return None
    covc = [r["coverage_cont"] for r in rows]
    reg = [r["reg_err_xy_m"] * 1000.0 for r in rows if r.get("reg_err_xy_m") is not None]
    wall = [r["wall_s"] for r in rows]
    casts = [r.get("reg_casts", 0) for r in rows]
    n = float(len(rows))
    reg.sort()
    q90 = reg[int(0.9 * (len(reg) - 1))] if reg else float("nan")
    return {
        "n": len(rows),
        "transfer_success": round(sum(r["success"] for r in rows) / n, 4),
        "mean_coverage_cont": round(sum(covc) / n, 4),
        "median_coverage_cont": round(st.median(covc), 4),
        "min_coverage_cont": round(min(covc), 4),
        "reg_err_xy_mm_median": round(st.median(reg), 2) if reg else None,
        "reg_err_xy_mm_p90": round(q90, 2) if reg else None,
        "reg_err_xy_mm_max": round(max(reg), 2) if reg else None,
        "reg_err_xy_n": len(reg),
        "reg_ok_rate": round(sum(r["reg_ok"] for r in rows) / n, 4),
        "mean_casts": round(sum(casts) / n, 1),
        "max_casts": max(casts),
        "wall_s_per_ep": round(sum(wall) / n, 4),
        "harness_errors": sum(1 for r in rows if r.get("status") not in (None, "ok")),
    }


def main():
    """Walk the levels, print the matrix, and bracket the 0.70 crossings per suite/arm."""
    matrix, envelope = {}, {}
    hdr = (f"{'level':>10} {'suite':>10} {'arm':>7} {'succ':>6} {'covc':>7} {'minc':>7} "
           f"{'reg_med':>8} {'reg_p90':>8} {'reg_max':>8} {'regok':>6} {'casts':>7} {'wall':>7}")
    print(hdr)
    print("-" * len(hdr))
    for lvl in LEVELS + [PROBE]:
        f = ROOT / (f"N197_r310_s100_{lvl}.jsonl" if lvl != PROBE
                    else f"N197_r310b_s20_{PROBE}.jsonl")
        if not f.exists():
            print(f"{lvl:>10}  MISSING {f}")
            continue
        rows = arms(f)
        cmp_rows = [json.loads(ln) for ln in f.read_text().splitlines()
                    if json.loads(ln).get("record") == "compare"]
        matrix[lvl] = {"dose": {}, "frozen": {}, "compare": {}}
        for s in SUITES:
            for arm in ("dose", "frozen"):
                st_ = stats([r for r in rows[arm] if r["suite"] == s])
                if st_ is None:
                    continue
                matrix[lvl][arm][s] = st_
                print(f"{lvl:>10} {s:>10} {arm:>7} {st_['transfer_success']:>6.2f} "
                      f"{st_['mean_coverage_cont']:>7.4f} {st_['min_coverage_cont']:>7.4f} "
                      f"{str(st_['reg_err_xy_mm_median']):>8} {str(st_['reg_err_xy_mm_p90']):>8} "
                      f"{str(st_['reg_err_xy_mm_max']):>8} {st_['reg_ok_rate']:>6.2f} "
                      f"{st_['mean_casts']:>7.1f} {st_['wall_s_per_ep']:>7.3f}")
            if cmp_rows:
                c = {k: v for k, v in cmp_rows[0].get(s, {}).items()
                     if k in ("delta", "welch_p", "fisher_p", "mean_a", "mean_b", "succ_a", "succ_b")}
                c["keep"] = cmp_rows[0].get("keep", cmp_rows[0].get(s, {}).get("keep"))
                matrix[lvl]["compare"][s] = c
        if cmp_rows:
            print(f"{'':>10} compare keep(B)={cmp_rows[0].get('keep')} "
                  f"pose_noise={cmp_rows[0].get('pose_noise_cfg')}")
    print()
    for arm in ("dose", "frozen"):
        for s in SUITES:
            succ = [matrix[l][arm][s]["transfer_success"] if matrix.get(l, {}).get(arm, {}).get(s) else None
                    for l in LEVELS + [PROBE]]
            bracket = None
            for i in range(1, len(succ)):
                a, b = succ[i - 1], succ[i]
                if a is not None and b is not None and float(a) > 0.70 >= float(b):
                    bracket = [LEVELS[i - 1], LEVELS[i]]
            envelope[f"{arm}/{s}"] = {"succ_by_level": succ, "crosses_0.70_between": bracket}
        print(f"{arm:>7} " + "  ".join(
            f"{s}:{','.join('--' if v is None else f'{v:.2f}' for v in envelope[f'{arm}/{s}']['succ_by_level'])}"
            for s in SUITES))
    print()
    for k, v in envelope.items():
        print(f"{k:>16}  sigma cross bracket: {v['crosses_0.70_between']}")
    (ROOT / "N197_matrix.json").write_text(json.dumps(
        {"levels": LEVELS, "matrix": matrix, "envelope": envelope}, indent=1) + "\n")
    print(f"\nwrote {ROOT / 'N197_matrix.json'}")


if __name__ == "__main__":
    main()
