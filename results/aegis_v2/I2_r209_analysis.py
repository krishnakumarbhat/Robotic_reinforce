"""Run 216: T209 = robust band-local soft-score, pooled-0, no hard gates (director iter 34).

Director iter 34 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: T207 lineage (band-local soft-score, pooled-0, no hard gates).
BASE: Run 214 (T207) = 60.0pts best of last 3; gated/shrunk (T206/T208) both 46.67 discard.
WHY: mean/sig + J95 fragile in thin bands; P90 auto-admit + w19 auto-reject +
  40/20 shrink all diluted local signal.
Variation vs T207: s=(w191-med_b)/(1.4826*MAD_b+eps) - 0.5*clip(jerk/J90_b,0,1).
  med/MAD robust to thin-band outliers; J90 (not J95) caps penalty earlier;
  lam FIXED 0.5 (no lam sweep); zero hard gates (no TAU_hi/lo).
  Global only as tie-break: if |s - TAU*_band| < 0.05, admit iff s_g >= TAU*_g
  (s_g = global robust score, TAU*_g global-fit; both train-locked).
KEEP: pooled-0, CAL on, zero hard gates.
Fit (train-locked, CALIB pooled only, single held eval):
  bands = CALIB jerk tertiles (qceil 1/3, 2/3); per-band med_b/MAD_b/J90_b;
  per-band ROC: TAU*_band over CALIB in-band fail-s quantiles Q{0.5..0.9}
  (qceil, deduped); per-band pick recall_b>=0.65 then min in-band pf,
  loosest (smallest TAU) tie-break. Global TAU*_g same objective on CALIB pooled.
Validation (all train-locked, single held eval, same bands pooled-0):
  ablate lam 0 vs 0.5 (s without jerk term, per-band refit);
  ablate MAD vs sig (T207-style (w-mu)/sig - 0.5*clip(jerk/J90,0,1), refit);
  ablate J90 vs J95 (robust z - 0.5*clip(jerk/J95_b,0,1), refit).
Kill: DISCARD iff pooled recall pts <= 48.67 (46.67+2) OR heldB keep < 60;
  director legs reported separately (pts>=70, heldB keep>=70).
  First KEEP -> robust line lives; else kill robust line.
Predicted 71-74pts, P(keep>=70)~0.55.
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
LAM = 0.5
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
TIE = 0.05
MAD_SCALE = 1.4826
MAD_EPS = 1e-9
PLATEAU = 46.67
KILL_PTS = PLATEAU + 2.0
KEEP_BAR = 60.0


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
cj_cal = sorted(e["jerk"] for e in cal_eps)
assert abs(qceil(cj_cal, 0.95) - EXP_J95) < 1e-6


def t173(e):
    return e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP


def w191(e):
    if not t173(e):
        return 0.0
    return min(1.0 / (max(e["slip_m"], FLOOR) + EPS), C99)


# ---- bands: CALIB jerk tertiles (train-locked, same as T207) ----
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
cal_fails = [e for e in cal_eps if not e["success"]]

# ---- per-band robust params (pooled-0) ----
med_band, mad_band, scale_band, J90_band, J95_band, mu_band, sig_band = ([] for _ in range(7))
for b in range(3):
    ws = [w191(e) for e in cal_band[b]]
    js = [e["jerk"] for e in cal_band[b]]
    med = median(ws)
    md = mad(ws, med)
    sc = MAD_SCALE * md + MAD_EPS
    if not (sc > 1e-12):
        sc = 1.0
    med_band.append(med)
    mad_band.append(md)
    scale_band.append(sc)
    J90_band.append(qceil(js, 0.9))
    J95_band.append(qceil(js, 0.95))
    mu = st.mean(ws)
    try:
        sig = st.stdev(ws)
    except Exception:
        sig = 0.0
    if not (sig > 1e-9):
        sig = 1.0
    mu_band.append(mu)
    sig_band.append(sig)

# ---- global robust params (tie-break only) ----
w_all = [w191(e) for e in cal_eps]
med_g = median(w_all)
mad_g = mad(w_all, med_g)
scale_g = MAD_SCALE * mad_g + MAD_EPS
if not (scale_g > 1e-12):
    scale_g = 1.0
J90_g = qceil(cj_cal, 0.9)


def clip01(x):
    return min(max(x, 0.0), 1.0)


def s_rob(e, lam=LAM):
    b = band_of(e)
    return (w191(e) - med_band[b]) / scale_band[b] - lam * clip01(e["jerk"] / J90_band[b])


def s_rob_lam0(e):
    b = band_of(e)
    return (w191(e) - med_band[b]) / scale_band[b]


def s_sig(e, lam=LAM):
    b = band_of(e)
    return (w191(e) - mu_band[b]) / sig_band[b] - lam * clip01(e["jerk"] / J90_band[b])


def s_j95(e, lam=LAM):
    b = band_of(e)
    return (w191(e) - med_band[b]) / scale_band[b] - lam * clip01(e["jerk"] / J95_band[b])


def s_g(e, lam=LAM):
    return (w191(e) - med_g) / scale_g - lam * clip01(e["jerk"] / J90_g)


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
    """Per-band TAU* fit on CALIB in-band fail scores (pooled-0)."""
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


def fit_global(score_fn):
    sc_fail = sorted(score_fn(e) for e in cal_fails)
    vals = []
    for q in Q_GRID:
        v = qceil(sc_fail, q)
        if not vals or abs(v - vals[-1]) > 1e-12:
            vals.append(v)
    best = None
    for tv in vals:
        s = pooled_stats(lambda e, tv=tv: score_fn(e) >= tv, cal_eps)
        if s["recall"] >= 0.65:
            key = (s["pf"], tv)
            if best is None or key < best[0]:
                best = (key, tv, s)
    return {"TAU": best[1] if best else None,
            "calib_recall": round(best[2]["recall"], 4) if best else None,
            "calib_pf": round(best[2]["pf"], 4) if best else None,
            "feasible": best is not None}


tau_main = fit_perband(s_rob)
g_pick = fit_global(s_g)
TAU_G = g_pick["TAU"]
feas_main = all(tau_main[b]["feasible"] for b in range(3)) and g_pick["feasible"]
TAUS = {b: tau_main[b]["TAU"] for b in range(3)} if feas_main else None


def admit_main(e):
    s = s_rob(e)
    t = TAUS[band_of(e)]
    if abs(s - t) < TIE:
        return s_g(e) >= TAU_G
    return s >= t


# ---- ablations (same bands, pooled-0, per-band refit) ----
tau_lam0 = fit_perband(s_rob_lam0)
tau_sig = fit_perband(s_sig)
tau_j95 = fit_perband(s_j95)


def mk_admit(tau_d, score_fn):
    if not all(tau_d[b]["feasible"] for b in range(3)):
        return (lambda e: False), False
    _t = {b: tau_d[b]["TAU"] for b in range(3)}
    return (lambda e, _t=_t: score_fn(e) >= _t[band_of(e)]), True


A_lam0, feas_lam0 = mk_admit(tau_lam0, s_rob_lam0)
A_sig, feas_sig = mk_admit(tau_sig, s_sig)
A_j95, feas_j95 = mk_admit(tau_j95, s_j95)

A = admit_main if feas_main else (lambda e: False)
a205 = lambda e: w191(e) >= T205_TAU  # noqa: E731
a199 = lambda e: w191(e) >= EXP_TAU50  # noqa: E731

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


e209 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = {"T199": pooled_eval(a199), "T205": pooled_eval(a205),
      "T207_local_ref": {"n_adm": 43, "score_recall_pts": 60.0},
      "T209": pooled_eval(A),
      "T209_lam0": pooled_eval(A_lam0),
      "T209_sig": pooled_eval(A_sig),
      "T209_J95": pooled_eval(A_j95)}
pts = pe["T209"]["score_recall_pts"]
kp = e209["keep_pct"]
tie_uses = sum(1 for e in tst_eps if abs(s_rob(e) - TAUS[band_of(e)]) < TIE) if feas_main else 0

held_fails = [e for e in tst_eps if not e["success"]]
rescued = sum(1 for e in held_fails if (not a205(e)) and (not A(e)))
new_mist = sum(1 for e in held_fails if (not a205(e)) and A(e))
rescue_rate = round(rescued / len(held_fails), 4) if held_fails else 0.0


def overlap(fn_a, fn_b, eps):
    sa = {id(e) for e in eps if fn_a(e)}
    sb = {id(e) for e in eps if fn_b(e)}
    inter, union = len(sa & sb), len(sa | sb)
    agree = sum(1 for e in eps if fn_a(e) == fn_b(e)) / len(eps)
    return {"agreement": round(agree, 4),
            "jaccard": round(inter / union, 4) if union else 1.0,
            "n_a": len(sa), "n_b": len(sb), "n_both": inter}


ov_held = {"T209_vs_T205": overlap(A, a205, tst_eps),
           "T209_vs_T199": overlap(A, a199, tst_eps)}

# per-band held recall at locked TAU*
band_roc_held = {}
for b in range(3):
    hb = [e for e in tst_eps if band_of(e) == b]
    hf = [e for e in hb if not e["success"]]
    if not hf or not feas_main:
        band_roc_held[b] = {"n": len(hb), "n_fails": len(hf)}
        continue
    t = TAUS[b]
    caught = sum(1 for e in hf if not (s_rob(e) >= t))
    band_roc_held[b] = {"n": len(hb), "n_fails": len(hf),
                        "TAU_star": round(t, 4),
                        "recall_band": round(caught / len(hf), 4)}

kill = bool((pts <= KILL_PTS) or (kp < KEEP_BAR) or (not feas_main))

out = {
    "variant": "T209 = robust band-local s=(w191-med_b)/(1.4826*MAD_b+eps)-0.5*clip(jerk/J90_b,0,1), pooled-0, no hard gates, global tie-break |d|<0.05",
    "frozen": {"TAU50_global_ref": EXP_TAU50, "theta_marg_bitident": True,
               "C99": round(C99, 4), "C99_degenerate": True,
               "J95_global_ref": EXP_J95, "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7),
               "w191_T173_embedded_kept": True,
               "no_TAUhi_lo_hard_gates": True,
               "fit_scope": "CALIB pooled only, jerk-tertile bands, single held eval, lam FIXED 0.5"},
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [len(x) for x in cal_band],
              "cal_fails": [len(x) for x in cal_fails_band],
              "med_band": [round(v, 4) for v in med_band],
              "MAD_band": [round(v, 4) for v in mad_band],
              "scale_band": [round(v, 4) for v in scale_band],
              "J90_band": [round(v, 6) for v in J90_band],
              "J95_band": [round(v, 6) for v in J95_band],
              "mu_band_T207ref": [round(v, 4) for v in mu_band],
              "sig_band_T207ref": [round(v, 4) for v in sig_band],
              "med_global": round(med_g, 4), "MAD_global": round(mad_g, 4),
              "scale_global": round(scale_g, 4), "J90_global": round(J90_g, 6)},
    "fit": {"LAM_FIXED": LAM, "TIE": TIE,
            "tau_star_per_band": {str(b): tau_main[b] for b in range(3)},
            "TAU_STAR_band": ({str(b): round(TAUS[b], 4) for b in range(3)} if TAUS else None),
            "global_tiebreak": {"TAU_G": round(TAU_G, 4) if TAU_G is not None else None,
                                "calib_recall": g_pick["calib_recall"],
                                "calib_pf": g_pick["calib_pf"],
                                "feasible": g_pick["feasible"],
                                "tie_uses_held": tie_uses},
            "abl_feasible": {"lam0": feas_lam0, "sig": feas_sig, "J95": feas_j95},
            "abl_tau": {"lam0": {str(b): tau_lam0[b] for b in range(3)},
                        "sig": {str(b): tau_sig[b] for b in range(3)},
                        "J95": {str(b): tau_j95[b] for b in range(3)}}},
    "perband_ROC_held": band_roc_held,
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T209": e209, "ref_T205": heldB_eval(a205), "cov_all": cov_all,
              "yield_vs_T205_pts": round(100 * (e209["yield_sel"] - heldB_eval(a205)["yield_sel"]), 2)},
    "pooled_held": {**pe,
                    "recall_vs_T199_pts": round(pe["T209"]["score_recall_pts"] - pe["T199"]["score_recall_pts"], 2),
                    "recall_vs_T205_pts": round(pe["T209"]["score_recall_pts"] - pe["T205"]["score_recall_pts"], 2),
                    "recall_vs_T207_pts": round(pe["T209"]["score_recall_pts"] - 60.0, 2),
                    "lam_vs_lam0_pts": round(pe["T209"]["score_recall_pts"] - pe["T209_lam0"]["score_recall_pts"], 2),
                    "MAD_vs_sig_pts": round(pe["T209"]["score_recall_pts"] - pe["T209_sig"]["score_recall_pts"], 2),
                    "J90_vs_J95_pts": round(pe["T209"]["score_recall_pts"] - pe["T209_J95"]["score_recall_pts"], 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["T209"]["p_fail_adm"]), 2)},
    "rescue_vs_T205": {"rescued": rescued, "new_mistakes": new_mist, "rescue_rate": rescue_rate},
    "admit_overlap_held": ov_held,
    "archive_adm_frac_T209": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e209["cov_adm"] >= cov_all - 0.02,
    "kill_rule": {"pts_le_48_67": bool(pts <= KILL_PTS), "keep_lt_60": bool(kp < KEEP_BAR),
                  "plateau": PLATEAU, "KILL_PTS": KILL_PTS, "KEEP_BAR": KEEP_BAR},
    "director_legs": {"pts_ge_70": bool(pts >= 70), "heldB_keep_ge_70": bool(kp >= 70),
                      "predicted_pts_71_74": pts, "predicted_keep": kp},
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "pts_gt_48_67": pts > KILL_PTS,
    "HELD_keep_ge_60": kp >= KEEP_BAR,
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T209"] == 0.0}
out["keep"] = bool(out["verdict_rule"]["pts_gt_48_67"] and out["verdict_rule"]["HELD_keep_ge_60"]
                   and out["runA_keep_cited"] and out["cov_ok"])
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
out["robust_line"] = ("LIVES (first KEEP)" if out["keep"] else "KILLED per director fail-branch")
json.dump(out, open("results/aegis_v2/I2_r209_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
