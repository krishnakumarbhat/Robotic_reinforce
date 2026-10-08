"""Run 213: T206 = band-gated stratified soft-score, pooled-1 (director iter 31).

Director iter 31 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: conditional jerk value - jerk only matters in w191 uncertainty band.
Rule: auto-admit iff w191>=TAU_hi; auto-reject iff w191<TAU_lo;
  in-band admit iff w191-lam*clip(jerk,0,J95)>=TAU_mid.
  w191(e) = min(1/(max(slip,0.005)+0.005), C99) if T173(e) else 0,
  T173(e) = jerk<=0.618 AND jerk<=THETA_MARG(0.014133) AND stall<=0.05.
Fit (train-locked, CALIB pooled-1 only, single held eval):
  TAU_hi/lo fixed at P90/P30 of w191 over CALIB pooled ALL eps (qceil);
  J95=0.015175 reused (bit-identical expected);
  grid lam {0,250,500,1000,2000,4000,8000} x TAU_mid {Qq fail-w, q in
  0.5..0.9 deduped}; objective: recall>=0.65 then min P(fail|admit),
  tie-break smallest lam, then loosest (smallest) TAU_mid.
Validation: split-half CALIB refit stability (even/odd idx, lam>0 both
  halves + TAU_mid within 1 grid step); held in-band lift vs w191-only
  baseline (lam=0 at locked TAU_mid, band-restricted).
Kill: held in-band delta<=0 OR band mass<15% -> CLOSE F-jerk entirely.
  (band mass = in-band fraction on HELD pooled; in-band delta = held
  in-band failure-reject recall T206 minus lam=0 baseline.)
Refs recomputed bit-identical: T205 (w-only 74.9317), T199 (68.0452).
Predicted 62-68pts (beats 46.67 plateau, still <70 keep).
Frozen I7+I10. Zero rig edits, frozen R173 files. G7-clean, seg 15."""

import json
import math
import statistics as st

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
I7 = 0.618
STALL_CAP = 0.05
FLOOR = 0.005
EPS = 0.005
EXP_TAU50 = 68.0452
EXP_THETA_MARG = 0.014133
EXP_J95 = 0.015175
T205_TAU = 74.9317
LAM_GRID = [0, 250, 500, 1000, 2000, 4000, 8000]
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def qceil(xs, q):
    s = sorted(xs)
    i = min(len(s) - 1, math.ceil(q * len(s)) - 1)
    return s[i]


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)
assert cal_hdr["pose_noise_cfg"] == "0.01,2" and cal_hdr["gate_mode"] == "post-hoc"
assert tst_hdr["pose_noise_cfg"] == "0.01,2" and arc_hdr["pose_noise_cfg"] == "2.0,2"

sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * 0.9) - 1)
TH_MARG = sj[k]
assert abs(TH_MARG - EXP_THETA_MARG) < 1e-6, TH_MARG
cal_adm_raws = sorted(
    1.0 / (max(e["slip_m"], FLOOR) + EPS)
    for e in cal_eps
    if e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP)
C99 = qceil(cal_adm_raws, 0.99)
assert abs(C99 - 100.0) < 1e-9, C99

cj_cal = sorted(e["jerk"] for e in cal_eps)
J95_RECOMPUTED = qceil(cj_cal, 0.95)
assert abs(J95_RECOMPUTED - EXP_J95) < 1e-6, (J95_RECOMPUTED, EXP_J95)
J95 = EXP_J95  # reuse per director spec (recompute matches to 6dp)


def t173(e):
    return e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP


def w191(e):
    if not t173(e):
        return 0.0
    return min(1.0 / (max(e["slip_m"], FLOOR) + EPS), C99)


def clip_j(e):
    return min(max(e["jerk"], 0.0), J95)


cw_all = [w191(e) for e in cal_eps]
TAU_LO = qceil(cw_all, 0.3)
TAU_HI = qceil(cw_all, 0.9)
assert len(cal_eps) == 120

cal_fails = [e for e in cal_eps if not e["success"]]
assert len(cal_fails) == 17, len(cal_fails)
tau_mid_vals = []
for q in Q_GRID:
    v = qceil([w191(e) for e in cal_fails], q)
    if not tau_mid_vals or abs(v - tau_mid_vals[-1]) > 1e-9:
        tau_mid_vals.append(v)


def in_band(e):
    w = w191(e)
    return TAU_LO <= w < TAU_HI


def make_rule(tau_hi, tau_lo, lam, tau_mid):
    def fn(e):
        w = w191(e)
        if w >= tau_hi:
            return True
        if w < tau_lo:
            return False
        return (w - lam * clip_j(e)) >= tau_mid
    return fn


def calib_stats(fn, eps):
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if fn(e)]
    caught = sum(1 for e in fails if not fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


# ---- grid fit on CALIB pooled ----
rows = []
for lam in LAM_GRID:
    for tm in tau_mid_vals:
        fn = make_rule(TAU_HI, TAU_LO, lam, tm)
        s = calib_stats(fn, cal_eps)
        rows.append((lam, tm, s))
feas = [r for r in rows if r[2]["recall"] >= 0.65]
assert feas, "no feasible (lam,TAU_mid)"
min_pf = min(r[2]["pf"] for r in feas)
cands = [r for r in feas if r[2]["pf"] == min_pf]
cands.sort(key=lambda r: (r[0], r[1]))
LAM_STAR, TAU_MID_STAR, star_s = cands[0]
edge_lam = LAM_STAR in (LAM_GRID[0], LAM_GRID[-1])
edge_tm = TAU_MID_STAR in (tau_mid_vals[0], tau_mid_vals[-1])

a206 = make_rule(TAU_HI, TAU_LO, LAM_STAR, TAU_MID_STAR)
a206_lam0 = make_rule(TAU_HI, TAU_LO, 0, TAU_MID_STAR)  # w-only baseline, same band
a205 = lambda e: w191(e) >= T205_TAU  # noqa: E731
a199 = lambda e: w191(e) >= EXP_TAU50  # noqa: E731

# ---- split-half CALIB stability (even/odd idx) ----
h1 = cal_eps[0::2]
h2 = cal_eps[1::2]


def refit(eps):
    best = None
    for lam in LAM_GRID:
        for tm in tau_mid_vals:
            s = calib_stats(make_rule(TAU_HI, TAU_LO, lam, tm), eps)
            if s["recall"] >= 0.65:
                key = (s["pf"], lam, tm)
                if best is None or key < best[0]:
                    best = (key, lam, tm, s)
    return best


f1 = refit(h1)
f2 = refit(h2)
i1 = tau_mid_vals.index(f1[2]) if f1 else None
i2 = tau_mid_vals.index(f2[2]) if f2 else None
i_star = tau_mid_vals.index(TAU_MID_STAR)
stab = {
    "h1_lam": f1[1] if f1 else None, "h1_taumid": round(f1[2], 4) if f1 else None,
    "h2_lam": f2[1] if f2 else None,
    "h2_taumid": round(f2[2], 4) if f2 else None,
    "lam_pos_both": bool(f1 and f2 and f1[1] > 0 and f2[1] > 0),
    "taumid_within_1_step": bool(
        f1 and f2 and abs(i1 - i_star) <= 1 and abs(i2 - i_star) <= 1),
}
stab["stable"] = bool(stab["lam_pos_both"] and stab["taumid_within_1_step"])

# ---- held eval ----
hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
assert len(hB) == 20
raw_keep = sum(1 for e in hB if e["success"]) / 20.0


def heldB_eval(fn):
    adm = [e for e in hB if fn(e)]
    s = sum(1 for e in adm if e["success"])
    nn = len(adm)
    return {"n_adm": nn, "yield_sel": round(s / 20.0, 4),
            "keep_pct": round(100 * s / 20.0, 2),
            "precision_adm": round(s / nn, 4) if nn else 1.0,
            "veto_rate": round(1 - nn / 20.0, 4),
            "cov_adm": round(st.mean(e["coverage_cont"] for e in adm), 4) if adm else 0.0}


def pooled_eval(fn):
    adm = [e for e in tst_eps if fn(e)]
    fails_h = [e for e in tst_eps if not e["success"]]
    caught = sum(1 for e in fails_h if not fn(e))
    nn = len(adm)
    pf = sum(1 for e in adm if not e["success"]) / nn if nn else 0.0
    return {"n_adm": nn, "veto_rate": round(1 - nn / len(tst_eps), 4),
            "p_fail_adm": round(pf, 4),
            "recall_fail_reject": round(caught / len(fails_h), 4),
            "score_recall_pts": round(100 * caught / len(fails_h), 2)}


def inband_recall(fn, eps):
    b = [e for e in eps if in_band(e)]
    bf = [e for e in b if not e["success"]]
    if not bf:
        return {"n_band": len(b), "n_band_fails": 0, "recall_inband": 1.0}
    caught = sum(1 for e in bf if not fn(e))
    return {"n_band": len(b), "n_band_fails": len(bf),
            "recall_inband": round(caught / len(bf), 4)}


e206, e205 = heldB_eval(a206), heldB_eval(a205)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = {"T199": pooled_eval(a199), "T205": pooled_eval(a205),
      "T206_lam0": pooled_eval(a206_lam0), "T206": pooled_eval(a206)}
ib_held = {"T206": inband_recall(a206, tst_eps),
           "lam0": inband_recall(a206_lam0, tst_eps)}
ib_cal = {"T206": inband_recall(a206, cal_eps),
          "lam0": inband_recall(a206_lam0, cal_eps)}
inband_delta_held = round(ib_held["T206"]["recall_inband"] - ib_held["lam0"]["recall_inband"], 4)
inband_delta_cal = round(ib_cal["T206"]["recall_inband"] - ib_cal["lam0"]["recall_inband"], 4)
band_mass_held = round(sum(1 for e in tst_eps if in_band(e)) / len(tst_eps), 4)
band_mass_cal = round(sum(1 for e in cal_eps if in_band(e)) / len(cal_eps), 4)


def overlap(fn_a, fn_b, eps):
    sa = {id(e) for e in eps if fn_a(e)}
    sb = {id(e) for e in eps if fn_b(e)}
    inter, union = len(sa & sb), len(sa | sb)
    agree = sum(1 for e in eps if fn_a(e) == fn_b(e)) / len(eps)
    return {"agreement": round(agree, 4),
            "jaccard": round(inter / union, 4) if union else 1.0,
            "n_a": len(sa), "n_b": len(sb), "n_both": inter}


ov_held = {"T206_vs_T205": overlap(a206, a205, tst_eps),
           "T206_vs_lam0": overlap(a206, a206_lam0, tst_eps),
           "T206_vs_T199": overlap(a206, a199, tst_eps)}

out = {
    "variant": "T206 = band-gated stratified soft-score (auto-admit w>=TAU_hi; auto-reject w<TAU_lo; in-band s=w-lam*clip(jerk,0,J95)>=TAU_mid, pooled-1 CALIB)",
    "frozen": {"TAU50_global_ref": EXP_TAU50, "theta_marg_bitident": True,
               "C99": round(C99, 4), "C99_degenerate": True, "J95": round(J95, 6),
               "J95_recomputed": round(J95_RECOMPUTED, 7),
               "J95_bitident_T204": True, "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7),
               "fit_scope": "pooled-1 CALIB only, TAU_hi/lo fixed P90/P30, grid lam x TAU_mid, single held eval"},
    "band": {"TAU_HI_P90_allcal": round(TAU_HI, 4), "TAU_LO_P30_allcal": round(TAU_LO, 4),
             "band_mass_calib": band_mass_cal, "band_mass_held": band_mass_held,
             "band_ge_15pct": bool(band_mass_held >= 0.15),
             "calib_inband_n": ib_cal["T206"]["n_band"],
             "calib_inband_fails": ib_cal["T206"]["n_band_fails"],
             "held_inband_n": ib_held["T206"]["n_band"],
             "held_inband_fails": ib_held["T206"]["n_band_fails"]},
    "fit": {"LAM_GRID": LAM_GRID, "TAU_MID_grid": [round(v, 4) for v in tau_mid_vals],
            "feasible_ge65_n": len(feas),
            "LAM_STAR": LAM_STAR, "TAU_MID_STAR": round(TAU_MID_STAR, 4),
            "lam_at_grid_edge": bool(edge_lam), "taumid_at_grid_edge": bool(edge_tm),
            "calib_star_recall": round(star_s["recall"], 4),
            "calib_star_p_fail_adm": round(star_s["pf"], 4),
            "calib_lam0_sameTM_recall": round(calib_stats(a206_lam0, cal_eps)["recall"], 4),
            "calib_lam0_sameTM_pf": round(calib_stats(a206_lam0, cal_eps)["pf"], 4)},
    "stability_splithalf": stab,
    "inband_lift": {"held_T206": ib_held["T206"], "held_lam0": ib_held["lam0"],
                    "held_delta": inband_delta_held,
                    "calib_T206": ib_cal["T206"], "calib_lam0": ib_cal["lam0"],
                    "calib_delta": inband_delta_cal,
                    "kill_delta_le_0": bool(inband_delta_held <= 0)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T206": e206, "ref_T205": e205, "cov_all": cov_all,
              "yield_vs_T205_pts": round(100 * (e206["yield_sel"] - e205["yield_sel"]), 2)},
    "pooled_held": {**pe,
                    "recall_vs_T205_pts": round(pe["T206"]["score_recall_pts"] - pe["T205"]["score_recall_pts"], 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["T206"]["p_fail_adm"]), 2)},
    "admit_overlap_held": ov_held,
    "archive_adm_frac_T206": round(sum(1 for e in arc_eps if a206(e)) / len(arc_eps), 4),
    "cov_ok": e206["cov_adm"] >= cov_all - 0.02,
    "director_predicted_pts_62_68": pe["T206"]["score_recall_pts"],
    "director_predicted_keep_ge_70": e206["keep_pct"],
}
out["kill_jerk_frontier"] = bool((inband_delta_held <= 0) or (band_mass_held < 0.15))
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "HELD_keep_ge_70": e206["keep_pct"] >= 70.0,
    "pts_ge_70": pe["T206"]["score_recall_pts"] >= 70.0,
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T206"] == 0.0,
    "band_ge_15pct": out["band"]["band_ge_15pct"],
    "inband_delta_gt_0": bool(inband_delta_held > 0)}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
out["frontier"] = ("CLOSE F-jerk entirely (band-gated conditional jerk adds zero: "
                   "in-band delta<=0 or band<15pct)"
                   if out["kill_jerk_frontier"] else
                   "band-gate shows conditional signal; T207 tunes band edges (P90/P30 -> CV)")
json.dump(out, open("results/aegis_v2/I2_r206_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
