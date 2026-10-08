"""Run 218: T211 = band-local upper-tail shrunk-scale soft-score, pooled-0 (director iter 36).

Director iter 36 (single-brain fallback muse-spark-1.3-contributor-free).
FRONTIER: keep T210 upper-tail multiplicative dampening, fix MAD-collapse + band-noise.
Rule (pre-reg, single held eval):
  s = max(0,w191-med_b)/(1.4826*MAD_shrunk+eps) * exp(-0.5*clip(jerk/J90_b,0,3))
  MAD_shrunk = max(MAD_b, F*MAD_global), F*=0.2; MAD_global = median_CALIB|w191-med_global|.
VARIATION vs T210: location stays pure local (no mu_shrunk washout from T208),
  scale gets global floor; additive pen (T209/T208) -> multiplicative exp damp, no lam.
VARIATION vs T209: keep median, replace -0.5*clip with exp damp, no lam to tune.
Fit (train-locked, CALIB pooled only): bands = CALIB jerk tertiles;
  per-band med_b/MAD_b/J90_b + MAD_global; per-band ROC TAU*_band over CALIB
  in-band fail-s quantiles Q{0.5..0.9} (qceil, deduped); recall_b>=0.65 then min
  in-band pf, loosest (smallest TAU) tie-break.
Validation: pooled-0 only; sweep MAD floor F{0.1,0.2,0.3} + damp D{0.5,1.0};
  ablate max(0,.) vs raw (no upper-tail clip). High-jerk slice = band-2 held
  recall + above-calib-median-jerk held-fail recall vs T210-additive refit (60.0).
Kill: DISCARD iff pooled recall pts < 65 OR no high-jerk gain vs additive refit
  (band2 held recall <= additive) OR needs lam retune (degenerate/infeasible).
Director legs: predicted 110-120pts (capped at 100 scale; treat as >=70),
  keep>=70, keep prob ~0.65, beats T210 on flat-band false fires.
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
T205_TAU = 74.9317
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
MAD_SCALE = 1.4826
MAD_EPS = 1e-9
J_EPS = 1e-12
FLOOR_STAR = 0.2
DAMP_STAR = 0.5
KILL_PTS = 65.0
KEEP_BAR = 70.0


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


def median(xs):
    return st.median(xs)


def mad(xs, med):
    return st.median([abs(x - med) for x in xs])


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)

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


def t173(e):
    return e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP


def w191(e):
    if not t173(e):
        return 0.0
    return min(1.0 / (max(e["slip_m"], FLOOR) + EPS), C99)


cj_cal = sorted(e["jerk"] for e in cal_eps)
E1 = qceil(cj_cal, 1.0 / 3.0)
E2 = qceil(cj_cal, 2.0 / 3.0)
assert E1 < E2


def band_of(e):
    j = e["jerk"]
    if j <= E1:
        return 0
    if j <= E2:
        return 1
    return 2


cal_band = [[e for e in cal_eps if band_of(e) == b] for b in range(3)]
assert all(len(x) == 40 for x in cal_band), [len(x) for x in cal_band]
cal_fails_band = [[e for e in cal_band[b] if not e["success"]] for b in range(3)]

w_cal_all = [w191(e) for e in cal_eps]
med_global = median(w_cal_all)
MAD_global = mad(w_cal_all, med_global)
assert MAD_global > 1e-12, MAD_global

med_band, MAD_band, J90_band = [], [], []
for b in range(3):
    ws = [w191(e) for e in cal_band[b]]
    js = [e["jerk"] for e in cal_band[b]]
    med = median(ws)
    md = mad(ws, med)
    med_band.append(med)
    MAD_band.append(md)
    J90_band.append(qceil(js, 0.9))
assert all(j > J_EPS for j in J90_band), J90_band


def shrunk_scale(b, F):
    return MAD_SCALE * max(MAD_band[b], F * MAD_global) + MAD_EPS


def clip03(x):
    return min(max(x, 0.0), 3.0)


def s_main(e, F=FLOOR_STAR, D=DAMP_STAR, upper=True):
    b = band_of(e)
    sc = shrunk_scale(b, F)
    num = max(0.0, w191(e) - med_band[b]) if upper else (w191(e) - med_band[b])
    return (num / sc) * math.exp(-D * clip03(e["jerk"] / J90_band[b]))


def s_add_refit(e, lam=0.5):
    b = band_of(e)
    sc = MAD_SCALE * MAD_band[b] + MAD_EPS
    if not (sc > 1e-12):
        sc = 1.0
    return (w191(e) - med_band[b]) / sc - lam * min(max(e["jerk"] / J90_band[b], 0.0), 1.0)


def pooled_stats(admit_fn, eps):
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if admit_fn(e)]
    caught = sum(1 for e in fails if not admit_fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


def inband_stats(admit_fn, eps_b):
    fails = [e for e in eps_b if not e["success"]]
    adm = [e for e in eps_b if admit_fn(e)]
    if not fails:
        return {"recall": 1.0, "pf": 0.0, "n_adm": len(adm)}
    caught = sum(1 for e in fails if not admit_fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


def fit_perband(score_fn):
    tau_star = {}
    for b in range(3):
        sc_fail = sorted(score_fn(e) for e in cal_fails_band[b])
        if not sc_fail:
            tau_star[b] = {"TAU": None, "feasible": False}
            continue
        vals = []
        for q in Q_GRID:
            v = qceil(sc_fail, q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        best = None
        for tv in vals:
            s = inband_stats(lambda e, tv=tv: score_fn(e) >= tv, cal_band[b])
            if s["recall"] >= 0.65:
                key = (s["pf"], tv)
                if best is None or key < best[0]:
                    best = (key, tv, s)
        tau_star[b] = {"TAU": best[1] if best else None,
                       "calib_recall_b": round(best[2]["recall"], 4) if best else None,
                       "calib_pf_b": round(best[2]["pf"], 4) if best else None,
                       "feasible": best is not None}
    return tau_star


def mk_admit(tau_d, score_fn):
    if not all(tau_d[b]["feasible"] for b in range(3)):
        return (lambda e: False), False
    _t = {b: tau_d[b]["TAU"] for b in range(3)}
    return (lambda e, _t=_t: score_fn(e) >= _t[band_of(e)]), True


variants = {
    "main_F02_D05_upper": lambda e: s_main(e, 0.2, 0.5, True),
    "F01_D05_upper": lambda e: s_main(e, 0.1, 0.5, True),
    "F03_D05_upper": lambda e: s_main(e, 0.3, 0.5, True),
    "F02_D10_upper": lambda e: s_main(e, 0.2, 1.0, True),
    "F02_D05_raw": lambda e: s_main(e, 0.2, 0.5, False),
}
tau_fit, adm_fit, feas_fit = {}, {}, {}
for name, fn in variants.items():
    t = fit_perband(fn)
    a, f = mk_admit(t, fn)
    tau_fit[name], adm_fit[name], feas_fit[name] = t, a, f

tau_add = fit_perband(s_add_refit)
A_add, feas_add = mk_admit(tau_add, s_add_refit)

A = adm_fit["main_F02_D05_upper"]
feas_main = feas_fit["main_F02_D05_upper"]
a205 = lambda e: w191(e) >= T205_TAU  # noqa: E731
a199 = lambda e: w191(e) >= EXP_TAU50  # noqa: E731

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


e211 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = {"T199": pooled_eval(a199), "T205": pooled_eval(a205),
      "T211_main": pooled_eval(A), "T211_add_refit": pooled_eval(A_add)}
for name in variants:
    if name != "main_F02_D05_upper":
        pe[name] = pooled_eval(adm_fit[name])
pts = pe["T211_main"]["score_recall_pts"]
kp = e211["keep_pct"]

cal_curve = {}
for name in list(variants) + ["add_refit"]:
    fn = adm_fit[name] if name in adm_fit else A_add
    c = pooled_stats(fn, cal_eps)
    cal_curve[name] = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in c.items()}

jmed_cal = median([e["jerk"] for e in cal_eps])


def slice_recall(fn, pred):
    sub = [e for e in tst_eps if (not e["success"]) and pred(e)]
    if not sub:
        return {"n_fails": 0, "recall": 1.0}
    caught = sum(1 for e in sub if not fn(e))
    return {"n_fails": len(sub), "recall": round(caught / len(sub), 4)}


hj = {}
for name in list(variants) + ["add_refit"]:
    fn = adm_fit[name] if name in adm_fit else A_add
    hj[name] = {
        "band2": slice_recall(fn, lambda e: band_of(e) == 2),
        "above_med": slice_recall(fn, lambda e: e["jerk"] > jmed_cal),
    }

# flat-band (band0, low jerk) false-fire check: admitted-fail rate in band0 held
flat = {}
for name in list(variants) + ["add_refit"]:
    fn = adm_fit[name] if name in adm_fit else A_add
    hb0 = [e for e in tst_eps if band_of(e) == 0]
    adm0 = [e for e in hb0 if fn(e)]
    f0 = sum(1 for e in adm0 if not e["success"])
    flat[name] = {"n_band0": len(hb0), "n_adm0": len(adm0),
                  "fail_adm0": f0,
                  "p_fail_adm0": round(f0 / len(adm0), 4) if adm0 else 0.0}

band_roc_held = {}
for b in range(3):
    hb = [e for e in tst_eps if band_of(e) == b]
    hf = [e for e in hb if not e["success"]]
    t = tau_fit["main_F02_D05_upper"][b]["TAU"] if feas_main else None
    if not hf or t is None:
        band_roc_held[b] = {"n": len(hb), "n_fails": len(hf)}
        continue
    caught = sum(1 for e in hf if not (variants["main_F02_D05_upper"](e) >= t))
    band_roc_held[b] = {"n": len(hb), "n_fails": len(hf),
                        "TAU_star": round(t, 4),
                        "recall_band_main": round(caught / len(hf), 4)}


def overlap(fn_a, fn_b, eps):
    agree = sum(1 for e in eps if fn_a(e) == fn_b(e)) / len(eps)
    sa = {id(e) for e in eps if fn_a(e)}
    sb = {id(e) for e in eps if fn_b(e)}
    inter, union = len(sa & sb), len(sa | sb)
    return {"agreement": round(agree, 4),
            "jaccard": round(inter / union, 4) if union else 1.0,
            "n_a": len(sa), "n_b": len(sb), "n_both": inter}


ov = {"main_vs_add": overlap(A, A_add, tst_eps),
      "main_vs_raw": overlap(A, adm_fit["F02_D05_raw"], tst_eps),
      "main_vs_T205": overlap(A, a205, tst_eps)}

gain_band2 = (hj["main_F02_D05_upper"]["band2"]["recall"] - hj["add_refit"]["band2"]["recall"]
              if feas_add else None)
kill_no_gain = (gain_band2 is None) or (gain_band2 <= 0)
needs_lam = False  # no lam in T211 by construction; set true only if infeasible fallback
kill = bool((pts < KILL_PTS) or kill_no_gain or (not feas_main))

out = {
    "variant": "T211 = max(0,w191-med_b)/(1.4826*max(MAD_b,0.2*MAD_global)+eps)*exp(-0.5*clip(jerk/J90_b,0,3)), pooled-0",
    "frozen": {"TAU50_global_ref": EXP_TAU50, "theta_marg_bitident": True,
               "C99": round(C99, 4), "C99_degenerate": True,
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7),
               "w191_T173_embedded_kept": True,
               "fit_scope": "CALIB pooled only, jerk-tertile bands, single held eval, no lam sweep, F*=0.2 D*=0.5"},
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [len(x) for x in cal_band],
              "cal_fails": [len(x) for x in cal_fails_band],
              "med_band": [round(v, 4) for v in med_band],
              "MAD_band": [round(v, 4) for v in MAD_band],
              "MAD_global": round(MAD_global, 4),
              "med_global": round(med_global, 4),
              "shrunk_scale_F02": [round(shrunk_scale(b, 0.2), 4) for b in range(3)],
              "raw_scale": [round(MAD_SCALE * MAD_band[b] + MAD_EPS, 4) for b in range(3)],
              "floor_binds_F02": [bool(MAD_band[b] < 0.2 * MAD_global) for b in range(3)],
              "J90_band": [round(v, 6) for v in J90_band],
              "jmed_cal": round(jmed_cal, 6)},
    "fit": {"TAU_main_per_band": {str(b): tau_fit["main_F02_D05_upper"][b] for b in range(3)},
            "TAU_add_per_band": {str(b): tau_add[b] for b in range(3)},
            "TAU_sweep": {n: {str(b): tau_fit[n][b] for b in range(3)} for n in variants},
            "feasible": {n: feas_fit[n] for n in variants} | {"add_refit": feas_add},
            "FLOOR_STAR": FLOOR_STAR, "DAMP_STAR": DAMP_STAR},
    "cal_curve_locked": cal_curve,
    "perband_ROC_held_main": band_roc_held,
    "high_jerk_slice_held": hj,
    "flat_band0_falsefire_held": flat,
    "high_jerk_gain_main_vs_add_band2": round(gain_band2, 4) if gain_band2 is not None else None,
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T211_main": e211, "ref_T205": heldB_eval(a205),
              "ref_add": heldB_eval(A_add),
              "cov_all": cov_all,
              "yield_vs_T205_pts": round(100 * (e211["yield_sel"] - heldB_eval(a205)["yield_sel"]), 2)},
    "pooled_held": {**pe,
                    "recall_vs_T199_pts": round(pe["T211_main"]["score_recall_pts"] - pe["T199"]["score_recall_pts"], 2),
                    "recall_vs_T205_pts": round(pe["T211_main"]["score_recall_pts"] - pe["T205"]["score_recall_pts"], 2),
                    "recall_vs_T207_60_pts": round(pe["T211_main"]["score_recall_pts"] - 60.0, 2),
                    "main_vs_add_pts": round(pe["T211_main"]["score_recall_pts"] - pe["T211_add_refit"]["score_recall_pts"], 2),
                    "main_vs_raw_pts": round(pe["T211_main"]["score_recall_pts"] - pe["F02_D05_raw"]["score_recall_pts"], 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["T211_main"]["p_fail_adm"]), 2)},
    "admit_overlap_held": ov,
    "archive_adm_frac_T211": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e211["cov_adm"] >= cov_all - 0.02,
    "kill_rule": {"pts_lt_65": bool(pts < KILL_PTS), "no_high_jerk_gain": bool(kill_no_gain),
                  "infeasible_main": bool(not feas_main),
                  "KILL_PTS": KILL_PTS, "KEEP_BAR": KEEP_BAR},
    "director_legs": {"pts_ge_70": bool(pts >= 70), "heldB_keep_ge_70": bool(kp >= KEEP_BAR),
                      "predicted_pts_110_120": pts, "predicted_keep": kp},
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "pts_ge_65": pts >= KILL_PTS,
    "high_jerk_gain": not kill_no_gain,
    "HELD_keep_ge_70": kp >= KEEP_BAR,
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T211"] == 0.0}
out["keep"] = bool(out["verdict_rule"]["pts_ge_65"] and out["verdict_rule"]["high_jerk_gain"]
                   and out["verdict_rule"]["HELD_keep_ge_70"]
                   and out["runA_keep_cited"] and out["cov_ok"])
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r211_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
