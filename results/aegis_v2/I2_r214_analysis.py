"""Run 221: T214 = band-local upper-tail EB-shrunk-scale soft-score pooled-0 (director iter 39).

Director iter 39 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: sample-size-aware scale shrinkage; T211-T213 tested fixed 0.2 floor /
  additive-jerk / global-mean blend only.
Variation: keep T211 structural winners (local med, upper-tail max(0,.),
  soft-score, pooled-0); change ONLY denominator: fixed floor -> n-weighted EB
  + 0.3 floor:
  s = max(0,w191-med_b)/(1.4826*max((n_b*MAD_b+10*MAD_glob)/(n_b+10),0.3*MAD_glob))
  No jerk term, no exp damp, no hard gates, no tie-break.
Fit (train-locked, CALIB pooled only, single held eval, pooled-0 only):
  bands = CALIB jerk tertiles (same edges as R218/T207-T213); per-band med_b/MAD_b
  + MAD_glob; EB scale_star_b; per-band ROC TAU*_band over CALIB in-band fail-s
  Q{0.5..0.9} (recall_b>=0.65 then min pf, loosest tie-break).
Validation: same bands as R218; ablation vs T211 (raw-MAD no-shrink refit);
  signed ablation (no max(0,.)); pooled-1 global refit of same s (leak check);
  thin-band variance + upper-tail FPR report; confirm signed still off.
Predicted 105-110pts keep (>=70) new BEST (director; recall caps at 100.0 so
  read as: feasible non-degenerate + beats T207 60.0 plateau + keep>=70).
Discard if: pooled recall <= T207 60.0 plateau, or degenerate all-reject,
  or pooled-0 vs pooled-1 gap closes (signals leak), or keep<70.
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
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
MAD_SCALE = 1.4826
MAD_EPS = 1e-9
K_EB = 10.0  # EB: (n_b*MAD_b + K*MAD_glob)/(n_b+K)
FLOOR_FRAC = 0.3  # max(EB, 0.3*MAD_glob)
PLATEAU = 46.67
T207_REF = 60.0


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


def t173(e):
    return e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP


def w191(e):
    if not t173(e):
        return 0.0
    return min(1.0 / (max(e["slip_m"], FLOOR) + EPS), C99)


E1 = qceil(cj_cal, 1.0 / 3.0)
E2 = qceil(cj_cal, 2.0 / 3.0)
assert E1 < E2, (E1, E2)


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

med_band, mad_band = [], []
for b in range(3):
    ws = [w191(e) for e in cal_band[b]]
    m = median(ws)
    med_band.append(m)
    mad_band.append(mad(ws, m))

w_all = [w191(e) for e in cal_eps]
MED_GLOB = median(w_all)
MAD_GLOB = mad(w_all, MED_GLOB)
n_band = [len(cal_band[b]) for b in range(3)]

# EB-shrunk scales (train-locked)
eb_raw, mad_star, scale_star, floor_binds = [], [], [], []
for b in range(3):
    eb = (n_band[b] * mad_band[b] + K_EB * MAD_GLOB) / (n_band[b] + K_EB)
    fl = FLOOR_FRAC * MAD_GLOB
    ms = max(eb, fl)
    eb_raw.append(eb)
    mad_star.append(ms)
    sc = MAD_SCALE * ms + MAD_EPS
    if not (sc > 1e-12):
        sc = 1.0
    scale_star.append(sc)
    floor_binds.append(eb < fl)
floor_frac = sum(floor_binds) / 3.0
collapse = [md < 1e-9 for md in mad_band]
collapse_frac = sum(collapse) / 3.0


def s_main(e):
    b = band_of(e)
    return max(0.0, w191(e) - med_band[b]) / scale_star[b]


def s_raw(e):  # ablation vs T211: raw MAD_b, no EB shrink
    b = band_of(e)
    sc = MAD_SCALE * mad_band[b] + MAD_EPS
    if not (sc > 1e-12):
        sc = 1.0
    return max(0.0, w191(e) - med_band[b]) / sc


def s_signed(e):  # signed ablation: no max(0,.), confirm still off
    b = band_of(e)
    return (w191(e) - med_band[b]) / scale_star[b]


def inband_stats(admit_fn, eps_b):
    fails = [e for e in eps_b if not e["success"]]
    adm = [e for e in eps_b if admit_fn(e)]
    if not fails:
        return {"recall": 1.0, "pf": 0.0, "n_adm": len(adm)}
    caught = sum(1 for e in fails if not admit_fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


def pooled_stats(admit_fn, eps):
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if admit_fn(e)]
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


def fit_global(score_fn, eps_pool, fails_pool):
    sc_fail = sorted(score_fn(e) for e in fails_pool)
    vals = []
    for q in Q_GRID:
        v = qceil(sc_fail, q)
        if not vals or abs(v - vals[-1]) > 1e-12:
            vals.append(v)
    best = None
    for tv in vals:
        s = pooled_stats(lambda e, tv=tv: score_fn(e) >= tv, eps_pool)
        if s["recall"] >= 0.65:
            key = (s["pf"], tv)
            if best is None or key < best[0]:
                best = (key, tv, s)
    if best is None:
        return {"TAU": None, "feasible": False}
    return {"TAU": best[1], "calib_recall": round(best[2]["recall"], 4),
            "calib_pf": round(best[2]["pf"], 4), "feasible": True}


tau_main = fit_perband(s_main)
feas_main = all(tau_main[b]["feasible"] for b in range(3))
TAUS = {b: tau_main[b]["TAU"] for b in range(3)} if feas_main else None
tau_raw = fit_perband(s_raw)
tau_signed = fit_perband(s_signed)
fit_p1 = fit_global(s_main, cal_eps, cal_fails)


def mk_admit(tau_d, score_fn):
    if not all(tau_d[b]["feasible"] for b in range(3)):
        return (lambda e: False), False
    _t = {b: tau_d[b]["TAU"] for b in range(3)}
    return (lambda e, _t=_t: score_fn(e) >= _t[band_of(e)]), True


A, feas = (mk_admit(tau_main, s_main) if feas_main else ((lambda e: False), False))
A_raw, feas_raw = mk_admit(tau_raw, s_raw)
A_signed, feas_signed = mk_admit(tau_signed, s_signed)
A_p1 = (lambda e: s_main(e) >= fit_p1["TAU"]) if fit_p1["feasible"] else (lambda e: False)
a205 = lambda e: w191(e) >= 74.9317  # noqa: E731
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


e214 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = {"T199": pooled_eval(a199), "T205": pooled_eval(a205),
      "T207_local_ref": {"n_adm": 43, "score_recall_pts": 60.0},
      "T214_pooled0": pooled_eval(A),
      "T214_rawMAD_abl": pooled_eval(A_raw),
      "T214_signed_abl": pooled_eval(A_signed),
      "T214_pooled1": pooled_eval(A_p1)}
pts = pe["T214_pooled0"]["score_recall_pts"]
pts_raw = pe["T214_rawMAD_abl"]["score_recall_pts"]
pts_signed = pe["T214_signed_abl"]["score_recall_pts"]
pts_p1 = pe["T214_pooled1"]["score_recall_pts"]
kp = e214["keep_pct"]
gap_p0_p1 = round(pts - pts_p1, 2)
leak = bool(abs(gap_p0_p1) < 5.0)

band_roc_held = {}
for b in range(3):
    hb = [e for e in tst_eps if band_of(e) == b]
    hf = [e for e in hb if not e["success"]]
    if not hf or not feas:
        band_roc_held[b] = {"n": len(hb), "n_fails": len(hf)}
        continue
    t = TAUS[b]
    caught = sum(1 for e in hf if not (s_main(e) >= t))
    adm_b = sum(1 for e in hb if s_main(e) >= t)
    band_roc_held[b] = {"n": len(hb), "n_fails": len(hf),
                        "TAU_star": round(t, 4),
                        "recall_band": round(caught / len(hf), 4),
                        "n_adm_band": adm_b}

# thin-band variance: which band has smallest n_fails / tightest MAD
thin_b = min(range(3), key=lambda b: (mad_star[b], len(cal_fails_band[b])))

out = {
    "variant": "T214 = band-local upper-tail EB-shrunk-scale soft-score pooled-0: s=max(0,w191-med_b)/(1.4826*max((n_b*MAD_b+10*MAD_glob)/(n_b+10),0.3*MAD_glob)), no jerk term",
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [len(x) for x in cal_band],
              "cal_fails": [len(x) for x in cal_fails_band],
              "med_band": [round(v, 4) for v in med_band],
              "MAD_band": [round(v, 4) for v in mad_band],
              "MAD_global": round(MAD_GLOB, 4),
              "MED_global": round(MED_GLOB, 4),
              "n_band": n_band,
              "K_EB": K_EB, "FLOOR_FRAC": FLOOR_FRAC,
              "EB_raw": [round(v, 4) for v in eb_raw],
              "MAD_star": [round(v, 4) for v in mad_star],
              "scale_star": [round(v, 4) for v in scale_star],
              "floor_binds": floor_binds, "floor_frac": round(floor_frac, 4),
              "collapse_raw_MAD": collapse, "collapse_frac": round(collapse_frac, 4),
              "thin_band": thin_b},
    "fit": {"tau_star_per_band": {str(b): tau_main[b] for b in range(3)},
            "TAU_STAR_band": ({str(b): round(TAUS[b], 4) for b in range(3)} if TAUS else None),
            "feasible": feas,
            "abl_rawMAD_feasible": feas_raw,
            "abl_rawMAD_tau": {str(b): tau_raw[b] for b in range(3)},
            "abl_signed_feasible": feas_signed,
            "abl_signed_tau": {str(b): tau_signed[b] for b in range(3)},
            "abl_pooled1": {"TAU": round(fit_p1["TAU"], 4) if fit_p1["TAU"] is not None else None,
                            "feasible": fit_p1["feasible"],
                            "calib_recall": fit_p1.get("calib_recall"),
                            "calib_pf": fit_p1.get("calib_pf")}},
    "perband_ROC_held": band_roc_held,
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T214": e214, "ref_T205": heldB_eval(a205), "cov_all": cov_all,
              "T214_rawMAD": heldB_eval(A_raw), "T214_signed": heldB_eval(A_signed),
              "T214_pooled1": heldB_eval(A_p1)},
    "pooled_held": {**pe,
                    "recall_vs_T207_pts": round(pts - T207_REF, 2),
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2),
                    "delta_main_vs_rawMAD_pts": round(pts - pts_raw, 2),
                    "delta_main_vs_signed_pts": round(pts - pts_signed, 2),
                    "pooled0_vs_pooled1_gap_pts": gap_p0_p1,
                    "leak_gap_closes_lt5": leak,
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["T214_pooled0"]["p_fail_adm"]), 2)},
    "archive_adm_frac_T214": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e214["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"pts_ge_70": bool(pts >= 70), "heldB_keep_ge_70": bool(kp >= 70),
                      "predicted_pts_105_110": pts, "predicted_keep": kp},
}
discard = (not feas or pts <= T207_REF or pts >= 100.0 - 1e-9 and pe["T214_pooled0"]["n_adm"] == 0
           or leak or kp < 70 or not out["cov_ok"])
# degenerate all-reject check explicit: n_adm==0 -> artifact, discard
if pe["T214_pooled0"]["n_adm"] == 0:
    discard = True
out["degenerate_all_reject"] = bool(pe["T214_pooled0"]["n_adm"] == 0)
out["verdict"] = "DISCARD" if discard else "KEEP"
out["keep"] = bool(out["verdict"] == "KEEP" and out["runA_keep_cited"])
json.dump(out, open("results/aegis_v2/I2_r214_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
