#!/usr/bin/env python3
"""N562 analysis -- attribution (D1-D5) + cost frontier probe (D6/D7) over the 9 paired cells.

Read-only: parses results/aegis_v2/N562_*.jsonl, prints per-cell candidate/baseline metrics,
the reg_diag pure observations, the compare record (Welch p coverage, Fisher p success, keep)
and the D1-D7 verdict lines. No coverage is computed here -- every number is read from the rig.
"""
import json
import math
import os
import sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "aegis_v2")
CELLS = [("A1_000", "0,0", "n=32"), ("A2_03264", "0.32,64", "n=32"),
         ("A3_096_n32", "0.96,192", "n=32"), ("A4_096_n128", "0.96,192", "n=128"),
         ("A5_096_fix0", "0.96,192", "n=32,POSE_FIX=0"),
         ("B1_160320", "1.60,320", "n=32"), ("B2_200400", "2.00,400", "n=32"),
         ("B3_240480", "2.40,480", "n=32"), ("B4_320640", "3.20,640", "n=32")]
SUITES = ["fixture_A", "fixture_B", "fixture_R"]
DIAG = ["reg_band_z_span", "reg_z_gap2", "reg_band_nbody", "reg_bbox_du_m", "reg_bbox_dv_m",
        "reg_est_gap_mm", "reg_err_du_mm", "reg_err_dv_mm", "reg_off_u_mm", "reg_off_v_mm",
        "reg_border_frac", "reg_pca_ratio", "reg_branch_margin_deg", "reg_branch_flips"]


def load(tag):
    p = os.path.join(D, f"N562_{tag}.jsonl")
    rows = [json.loads(x) for x in open(p) if x.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    summ = {(r["path_mode"], s): r for r in rows if r.get("record") == "summary_mode"
            for s in [None]}
    summ = {}
    for r in rows:
        if r.get("record") == "summary_mode":
            summ[r["path_mode"]] = r
    return hdr, eps, cmp_, summ


def pct(vals, q):
    xs = sorted(float(v) for v in vals if isinstance(v, (int, float))
                and math.isfinite(float(v)))
    if len(xs) <= 1:
        return xs[0] if xs else float("nan")
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


print("=" * 100)
print("N562  CELLS  (candidate = certified N478 stack;  baseline = --compare trochoid + "
      "--compare-env frozen single-cast default;  20 seeds, G4 paired)")
print("=" * 100)
for tag, pn, note in CELLS:
    hdr, eps, cmp_, summ = load(tag)
    cand_lbl = "trochoid"
    base_lbl = next(k for k in summ if k != cand_lbl) if len(summ) > 1 else None
    print(f"\n### {tag}  pose_noise={pn}  {note}   harness_errors="
          f"{next(s for s in summ.values()).get('harness_errors')}")
    print(f"    header: reg={hdr['aegis_reg']} H={hdr['aegis_reg_half_m']} n={hdr['aegis_reg_n']} "
          f"cn={hdr['aegis_reg_cn']} dfact={hdr['aegis_reg_dfact']} casts={hdr['aegis_reg_casts']} "
          f"row_centre={hdr['row_centre']}")
    print(f"    {'suite':10s} {'B_succ(c/b)':>13s} {'covc c':>8s} {'covc b':>8s} "
          f"{'d_covc':>8s} {'welch_p':>10s} {'fisher_p':>10s} {'regp50':>8s} {'regp90':>8s} "
          f"{'regok':>6s} {'esc':>6s} {'brd':>6s}")
    for s in SUITES:
        c = summ[cand_lbl]["per_suite"].get(s, {})
        b = summ[base_lbl]["per_suite"].get(s, {}) if base_lbl else {}
        cc = cmp_[s] if cmp_ else {}
        ce = [e for e in eps if e.get("suite") == s and e["path_mode"] == cand_lbl]
        nok = sum(1 for e in ce if e.get("reg_ok"))
        esc = sum(float(e.get("escaped_frac", 0.0)) for e in ce) / max(len(ce), 1)
        brd = sum(float(e.get("reg_diag_mean", {}).get("reg_border_frac", 0.0))
                  for e in ce) / max(len(ce), 1)
        print(f"    {s:10s} {c.get('succ_n','?')}/{b.get('succ_n','?'):>3}".replace("/", "/")
              + f" {'':0s}", end="")
        print()
        break
    # proper table
    print(f"    {'suite':10s} {'succ c/b':>10s} {'covc c':>8s} {'covc b':>8s} {'d_covc':>8s} "
          f"{'welch_p':>10s} {'fisher_p':>10s} {'regp50':>8s} {'regp90':>8s} {'regok':>6s} "
          f"{'esc':>6s}")
    for s in SUITES:
        c = summ[cand_lbl]["per_suite"].get(s, {})
        b = summ[base_lbl]["per_suite"].get(s, {}) if base_lbl else {}
        cc = cmp_[s] if cmp_ else {}
        ce = [e for e in eps if e.get("suite") == s and e["path_mode"] == cand_lbl]
        nok = sum(1 for e in ce if e.get("reg_ok"))
        esc = sum(float(e.get("escaped_frac", 0.0)) for e in ce) / max(len(ce), 1)
        wp = cc.get("welch_p")
        fp = cc.get("fisher_p")
        print(f"    {s:10s} {c.get('success', 0):.2f}/{b.get('success', 0):.2f}    "
              f"{c.get('coverage_cont', 0):8.4f} {b.get('coverage_cont', 0):8.4f} "
              f"{cc.get('delta', 0):+8.4f} "
              f"{(f'{wp:.3e}' if isinstance(wp, float) else 'na'):>10s} "
              f"{(f'{fp:.3e}' if isinstance(fp, float) else 'na'):>10s} "
              f"{c.get('reg_err_xy_p50_mm', float('nan')):8.2f} "
              f"{c.get('reg_err_xy_p90_mm', float('nan')):8.2f} {nok:3d}/{len(ce):<3d} "
              f"{esc:6.3f}")
    if cmp_:
        print(f"    COMPARE keep={cmp_['keep']}  metric={cmp_['metric']}  "
              f"pose_noise_cfg={cmp_['pose_noise_cfg']}")
    cs = summ[cand_lbl]["per_suite"]["fixture_B"].get("reg_diag_mean", {})
    print("    B reg_diag: " + "  ".join(
        f"{k}={cs.get(k, float('nan'))}" for k in DIAG if k in cs))

# ---- D1..D7 verdicts ------------------------------------------------------
print("\n" + "=" * 100)
print("PRE-REGISTERED FALSIFIERS (D1-D7, stated in ideas.md + rig source N561 block before any dose)")
print("=" * 100)


def bdiag(tag):
    _, _, _, summ = load(tag)
    return summ["trochoid"]["per_suite"]["fixture_B"]


def beps(tag):
    _, eps, _, _ = load(tag)
    return [e for e in eps if e.get("suite") == "fixture_B" and e["path_mode"] == "trochoid"]


def bsucc(tag):
    _, _, _, summ = load(tag)
    return summ["trochoid"]["per_suite"]["fixture_B"].get("success", 0.0)


# D1
a1, a3 = bdiag("A1_000"), bdiag("A3_096_n32")
spans = [e.get("reg_band_z_span") for e in beps("A3_096_n32")]
nbody = [e.get("reg_band_nbody") for e in beps("A3_096_n32")]
du = abs(a3.get("reg_bbox_du_m", 0.0))
dv = abs(a3.get("reg_bbox_dv_m", 0.0))
pitch = a3.get("reg_pitch_m", 0.0)
print(f"D1  cell (i) 1cm top-face band admits a rim/pedestal: "
      f"band_z_span max={max(x for x in spans if x is not None):.6f}  nbody set={sorted(set(nbody))}  "
      f"|bbox_du|={du:.4f} ({du/pitch:.2f} pitch)  |bbox_dv|={dv:.4f} ({dv/pitch:.2f} pitch)  "
      f"pitch={pitch:.5f}")
print(f"    ALIVE iff z_span>1e-4 OR nbody>1 OR |du|>pitch OR |dv|>pitch  ->  "
      f"{'ALIVE' if (max(x for x in spans if x is not None) > 1e-4 or max(nbody) > 1 or du > pitch or dv > pitch) else 'FALSIFIED'}")

# D2
mar = [e.get("reg_branch_margin_deg") for e in beps("A3_096_n32")]
fl = [e.get("reg_branch_flips") for e in beps("A3_096_n32")]
print(f"D2  cell (ii) PCA pi-branch margin: min={min(x for x in mar if x is not None):.2f} deg  "
      f"flips set={sorted(set(fl))}  -> "
      f"{'ALIVE' if (min(x for x in mar if x is not None) < 10 or max(fl) > 0) else 'FALSIFIED'}"
      f"  (pre-registered prediction was FALSIFIED)")

# D3
p50_32 = bdiag("A3_096_n32")["reg_err_xy_p50_mm"]
p50_128 = bdiag("A4_096_n128")["reg_err_xy_p50_mm"]
p90_32 = bdiag("A3_096_n32")["reg_err_xy_p90_mm"]
p90_128 = bdiag("A4_096_n128")["reg_err_xy_p90_mm"]
print(f"D3  estimator sampling term owns it iff p50 falls >=2x from n=32 to n=128: "
      f"p50 {p50_32:.2f} -> {p50_128:.2f} mm (ratio {p50_32/max(p50_128,1e-9):.2f}x)  "
      f"p90 {p90_32:.2f} -> {p90_128:.2f} mm  -> "
      f"{'CONFIRMED' if p50_32 >= 2 * p50_128 else 'REFUTED (residual left UNNAMED)'}")
print(f"    A arm: p50 {bdiag('A1_000')['reg_err_xy_p50_mm']:.2f} -> "
      f"{bdiag('A4_096_n128')['reg_err_xy_p50_mm'] if 'A' else ''}")

# D4
p50_fix = bdiag("A5_096_fix0")["reg_err_xy_p50_mm"]
print(f"D4  placement-not-geometry: with POSE_FIX_U/V=0 and the yaw dose kept, B p50 must "
      f"fall below 2 mm: {p50_32:.2f} -> {p50_fix:.2f} mm  -> "
      f"{'PLACEMENT OWNS IT' if p50_fix < 2.0 else 'REFUTED -- residual is ray/face geometry alone'}")

# D5
print("D5  no lever claimed from attribution: Phase A arms are the certified stack or its paired "
      "single-cast baseline; any coverage movement is reported as a paired certificate or not at all.")

# D6/D7
print("\nD6/D7  COST PROBE above the certified 1.60,320 wall (candidate arm, 20 seeds):")
print(f"    {'cell':>10s} {'B succ':>8s} {'covc':>8s} {'R succ':>8s} {'A succ':>8s} "
      f"{'regp50':>8s} {'regok':>7s} {'escB':>7s} {'launchB':>8s}")
for tag, pn, _ in CELLS[5:]:
    _, _, _, summ = load(tag)
    _, eps, _, _ = load(tag)
    pe = summ["trochoid"]["per_suite"]
    eb = [e for e in eps if e.get("suite") == "fixture_B" and e["path_mode"] == "trochoid"]
    nok = sum(1 for e in eb if e.get("reg_ok"))
    esc = sum(float(e.get("escaped_frac", 0.0)) for e in eb) / max(len(eb), 1)
    lau = sum(float(e.get("mean_launch_frac", e.get("launch_frac", 0.0))) for e in eb) / max(len(eb), 1)
    print(f"    {pn:>10s} {pe['fixture_B'].get('success',0):8.2f} "
          f"{pe['fixture_B'].get('coverage_cont',0):8.4f} {pe['fixture_R'].get('success',0):8.2f} "
          f"{pe['fixture_A'].get('success',0):8.2f} {pe['fixture_B'].get('reg_err_xy_p50_mm',float('nan')):8.2f} "
          f"{nok:3d}/{len(eb):<3d} {esc:7.3f} {lau:8.3f}")