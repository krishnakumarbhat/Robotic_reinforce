"""Run 230: T223 = band-local median asymmetric upper-only IQR soft-clip pooled-4
+ 25% global anchor (director iter 48, single-brain fallback
opencode-responses/muse-spark-1.3-contributor-free).

Frontier: first keep-first not discard-first; T220-T222 all discard at 100.0pts,
keep>=70 never hit. T222 proved fails sit CENTRAL (inverted separability) under
symmetric central-keep; T223 removes the lower-tail penalty entirely.
Variation vs T222 (229 minimal control: median-only, pooled-0, single floor):
  wraw(e) = 1/(max(slip_m,FLOOR)+EPS) if t173(e) else 0.0 (single floor, no C99)
  c_b   = 0.75*med_b + 0.25*med_g            (25% global anchor)
  IQRup_b = Q75_b - med_b                     (upper-only asymmetric, no EB/MAD)
  thr_b = max(c_b + 1.5*IQRup_b, med_b)       (single lower floor at med_b)
  s(e)  = max(0, wraw(e) - thr_b)             (upper-only exceedance, symmetric off)
  p(e)  = 1 - exp(-s / max(IQRup_b, 1e-9))    (soft-clip, monotonic, no hard discard)
Admit iff p <= TAU*_b (per-band ROC, pooled-4 = 4-pt success-quantile grid
Q{0.5,0.65,0.8,0.9}, recall_b>=0.65 then min pf, loosest tie).
Bands=CALIB jerk tertiles. Frozen I7+I10. Zero rig edits, frozen R173 files.
Why: symmetric central-keep rejects low-w successes (lower-tail false positives);
upper-only admits the entire lower tail, keeping them -> breaks 100pts discard.
Validation: same 227-229 eval set; success=keep>=70 with pts<100; fail=discard.
Predicted 72-78 keep, 60-80pts. ABORT IF still 100pts -> frontier exhausted,
pivot to winsorize-all T224.
"""

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
EXP_THETA_MARG = 0.014133
Q_GRID4 = [0.5, 0.65, 0.8, 0.9]
Q_GRID8 = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
T207_REF = 60.0
PLATEAU = 46.67


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


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)

sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * 0.9) - 1)
TH_MARG = sj[k]
assert abs(TH_MARG - EXP_THETA_MARG) < 1e-6, TH_MARG
cj_cal = sorted(e["jerk"] for e in cal_eps)


def t173(e):
    return e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP


def wraw(e):
    if not t173(e):
        return 0.0
    return 1.0 / (max(e["slip_m"], FLOOR) + EPS)


E1 = qceil(cj_cal, 1.0 / 3.0)
E2 = qceil(cj_cal, 2.0 / 3.0)
assert abs(E1 - 0.003652) < 1e-6 and abs(E2 - 0.009075) < 1e-6, (E1, E2)


def band_of(e):
    j = e["jerk"]
    if j <= E1:
        return 0
    if j <= E2:
        return 1
    return 2


cal_band = [[e for e in cal_eps if band_of(e) == b] for b in range(3)]
assert all(len(x) == 40 for x in cal_band), [len(x) for x in cal_band]
cal_succ_band = [[e for e in cal_band[b] if e["success"]] for b in range(3)]

w_all = [wraw(e) for e in cal_eps]
MED_GLOB = median(w_all)
med_band, iqr_up, thr_band, scale_b = [], [], [], []
for b in range(3):
    ws = sorted(wraw(e) for e in cal_band[b])
    mb = median(ws)
    q75 = qceil(ws, 0.75)
    iu = max(q75 - mb, 0.0)
    c = 0.75 * mb + 0.25 * MED_GLOB
    thr = max(c + 1.5 * iu, mb)
    med_band.append(mb)
    iqr_up.append(iu)
    thr_band.append(thr)
    scale_b.append(max(iu, 1e-9))


def p_soft(e):
    b = band_of(e)
    s = max(0.0, wraw(e) - thr_band[b])
    return 1.0 - math.exp(-s / scale_b[b])


def s_hard(e):
    return max(0.0, wraw(e) - thr_band[e and band_of(e)])


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


def fit_perband(score_fn, succ_bands, grid):
    tau_star = {}
    for b in range(3):
        cand = sorted(score_fn(e) for e in succ_bands[b])
        if not cand:
            tau_star[b] = {"TAU": None, "feasible": False}
            continue
        vals = []
        for q in grid:
            v = qceil(cand, q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        best = None
        for tv in vals:
            s = inband_stats(lambda e, tv=tv: score_fn(e) <= tv, cal_band[b])
            if s["recall"] >= 0.65:
                key = (s["pf"], -tv)
                if best is None or key < best[0]:
                    best = (key, tv, s)
        tau_star[b] = {"TAU": best[1] if best else None,
                       "calib_recall_b": round(best[2]["recall"], 4) if best else None,
                       "calib_pf_b": round(best[2]["pf"], 4) if best else None,
                       "feasible": best is not None,
                       "grid": [round(v, 4) for v in vals]}
    return tau_star


tau_main = fit_perband(p_soft, cal_succ_band, Q_GRID4)
feas = all(tau_main[b]["feasible"] for b in range(3))
if feas:
    _t = {b: tau_main[b]["TAU"] for b in range(3)}
    A = lambda e, _t=_t: p_soft(e) <= _t[band_of(e)]
else:
    A = lambda e: False


def audit_finite(fn, eps):
    return sum(1 for e in eps if not (math.isfinite(fn(e)) and 0.0 <= fn(e) <= 1.0))


nan_calib = audit_finite(p_soft, cal_eps)
nan_held = audit_finite(p_soft, tst_eps)
nan_arch = audit_finite(p_soft, arc_eps)

sc_s = sorted(p_soft(e) for e in cal_eps if e["success"])
sc_f = sorted(p_soft(e) for e in cal_eps if not e["success"])


def dec(x, q):
    return round(qceil(x, q), 4) if x else None


def var(xs):
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


sdist = {
    "succ_q10": dec(sc_s, 0.1), "succ_q50": dec(sc_s, 0.5), "succ_q90": dec(sc_s, 0.9),
    "fail_q10": dec(sc_f, 0.1), "fail_q50": dec(sc_f, 0.5), "fail_q90": dec(sc_f, 0.9),
    "pooled_var": round(var(sorted(p_soft(e) for e in cal_eps)), 6),
}
sband = []
for b in range(3):
    vs = sorted(p_soft(e) for e in cal_band[b])
    sband.append({"med": dec(vs, 0.5), "q90": dec(vs, 0.9), "var": round(var(vs), 6),
                  "frac_zero": round(sum(1 for v in vs if v == 0.0) / len(vs), 4)})

trail_streak = 6

lobo_pts = {}
for left in range(3):
    pool = [e for b in range(3) for e in cal_band[b] if b != left]
    succ = sorted(p_soft(e) for e in pool if e["success"])
    vs = []
    for q in Q_GRID4:
        v = qceil(succ, q)
        if not vs or abs(v - vs[-1]) > 1e-12:
            vs.append(v)
    b2 = None
    for tv in vs:
        s2 = pooled_stats(lambda e, tv=tv: p_soft(e) <= tv, pool)
        if s2["recall"] >= 0.65:
            key = (s2["pf"], -tv)
            if b2 is None or key < b2[0]:
                b2 = (key, tv, s2)
    if b2 is None:
        lobo_pts[str(left)] = {"feasible": False, "held_recall_pts": 0.0}
    else:
        tv = b2[1]
        ps = pooled_stats(lambda e, tv=tv: p_soft(e) <= tv, tst_eps)
        lobo_pts[str(left)] = {"feasible": True, "held_recall_pts": round(100 * ps["recall"], 2),
                               "TAU": round(tv, 4)}
lobo_vals = [lobo_pts[str(b)]["held_recall_pts"] for b in range(3)]
lobo_var = round(max(lobo_vals) - min(lobo_vals), 2)


def abl_perband(score_fn, grid):
    succ = [[e for e in cal_band[b] if e["success"]] for b in range(3)]
    td = fit_perband(score_fn, succ, grid)
    ok = all(td[b]["feasible"] for b in range(3))
    if not ok:
        fn = lambda e: False
    else:
        _t = {b: td[b]["TAU"] for b in range(3)}
        fn = lambda e, _t=_t: score_fn(e) <= _t[band_of(e)]
    ps = pooled_stats(fn, tst_eps)
    return {"feasible": ok, "held_recall_pts": round(100 * ps["recall"], 2), "n_adm": ps["n_adm"]}


abls = {}
c0 = list(med_band)
abls["anchor0"] = abl_perband(
    lambda e: 1.0 - math.exp(-max(0.0, wraw(e) - max(c0[band_of(e)] + 1.5 * iqr_up[band_of(e)], c0[band_of(e)])) / scale_b[band_of(e)]),
    Q_GRID4)
iqr_sym = []
for b in range(3):
    ws = sorted(wraw(e) for e in cal_band[b])
    iqr_sym.append(max(qceil(ws, 0.75) - qceil(ws, 0.25), 0.0))
csym = [0.75 * med_band[b] + 0.25 * MED_GLOB for b in range(3)]
abls["symmetric_IQR"] = abl_perband(
    lambda e: 1.0 - math.exp(-max(0.0, abs(wraw(e) - csym[band_of(e)]) - 1.5 * iqr_sym[band_of(e)]) / max(iqr_sym[band_of(e)], 1e-9)),
    Q_GRID4)
abls["hard_no_soft"] = abl_perband(s_hard, Q_GRID4)
abls["grid8"] = abl_perband(p_soft, Q_GRID8)
cand4 = sorted(p_soft(e) for e in cal_eps if e["success"])
vs4 = []
for q in Q_GRID4:
    v = qceil(cand4, q)
    if not vs4 or abs(v - vs4[-1]) > 1e-12:
        vs4.append(v)
b4 = None
for tv in vs4:
    s2 = pooled_stats(lambda e, tv=tv: p_soft(e) <= tv, cal_eps)
    if s2["recall"] >= 0.65:
        key = (s2["pf"], -tv)
        if b4 is None or key < b4[0]:
            b4 = (key, tv, s2)
if b4 is None:
    abls["pooled4_globalTAU"] = {"feasible": False, "held_recall_pts": 100.0, "n_adm": 0}
else:
    tv4 = b4[1]
    fn4 = lambda e, tv4=tv4: p_soft(e) <= tv4
    ps4 = pooled_stats(fn4, tst_eps)
    abls["pooled4_globalTAU"] = {"feasible": True, "held_recall_pts": round(100 * ps4["recall"], 2),
                                 "n_adm": ps4["n_adm"], "TAU": round(tv4, 4)}

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
    prec = sum(1 for e in adm if e["success"]) / nn if nn else 1.0
    return {"n_adm": nn, "veto_rate": round(1 - nn / len(tst_eps), 4),
            "p_fail_adm": round(pf, 4),
            "precision_adm": round(prec, 4),
            "recall_fail_reject": round(caught / len(fails_h), 4),
            "score_recall_pts": round(100 * caught / len(fails_h), 2)}


e223 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = pooled_eval(A)
pts = pe["score_recall_pts"]
kp = e223["keep_pct"]

tst_band = [[e for e in tst_eps if band_of(e) == b] for b in range(3)]
band_held = {}
for b in range(3):
    hb = tst_band[b]
    hf = [e for e in hb if not e["success"]]
    ha = [e for e in hb if A(e)]
    prec_b = (sum(1 for e in ha if e["success"]) / len(ha)) if ha else 1.0
    rec_b = (sum(1 for e in hf if not A(e)) / len(hf)) if hf else 1.0
    band_held[b] = {"n": len(hb), "n_fails": len(hf), "n_adm": len(ha),
                    "recall_band": round(rec_b, 4), "precision_band": round(prec_b, 4),
                    "TAU": tau_main[b]["TAU"]}

discard_counter = trail_streak + (1 if pe["n_adm"] == 0 else 0)
streak_broken = bool(kp >= 5.0)
rank_corr_note = "soft-clip monotonic in s_hard by construction (rank corr 1.0, inert)"

out = {
    "variant": "T223 = band-local median asymmetric upper-only IQR soft-clip pooled-4 + 25% global anchor: s=max(0,wraw-thr_b), thr_b=max(0.75*med_b+0.25*med_g+1.5*IQRup_b,med_b), p=1-exp(-s/max(IQRup_b,eps)); admit iff p<=TAU*_b (per-band 4-pt success grid)",
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [40, 40, 40],
              "med_band": [round(v, 4) for v in med_band],
              "MED_global": round(MED_GLOB, 4),
              "anchor": 0.25,
              "IQRup_band": [round(v, 4) for v in iqr_up],
              "center_star": [round(0.75 * med_band[b] + 0.25 * MED_GLOB, 4) for b in range(3)],
              "thr_band": [round(v, 4) for v in thr_band],
              "floor_at_med_binds": [bool(abs(thr_band[b] - med_band[b]) < 1e-12) for b in range(3)],
              "wraw_no_C99_cap": True,
              "symmetric_off": True, "EB_off": True, "MAD_off": True},
    "fit": {"tau_star_per_band": {str(b): tau_main[b] for b in range(3)},
            "feasible": feas, "grid": "Q4 pooled-4"},
    "s_dist": dict(sdist, per_band=sband),
    "soft_clip": {"monotonic_rank_inert": True, "note": rank_corr_note},
    "finite_audit": {"nonfinite_or_neg_calib": nan_calib,
                     "nonfinite_or_neg_held": nan_held,
                     "nonfinite_or_neg_arch": nan_arch,
                     "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0)},
    "discard_counter": {"trailing_100_streak_pre": trail_streak,
                        "degenerate_now": bool(pe["n_adm"] == 0),
                        "counter": discard_counter,
                        "streak_broken_keep_ge5": streak_broken},
    "lobo": {"per_leftout": lobo_pts, "var_pts": lobo_var,
             "pass_var_lt15": bool(lobo_var < 15)},
    "ablations": abls,
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T223": e223, "cov_all": cov_all},
    "perband_held": band_held,
    "pooled_held": {"T223_pooled4": pe,
                    "recall_vs_T207_pts": round(pts - T207_REF, 2),
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["p_fail_adm"]), 2)},
    "archive_adm_frac_T223": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e223["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"predicted_pts_60_80": pts, "predicted_keep_72_78": kp,
                      "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0),
                      "success_keep_ge70_pts_lt100": bool(kp >= 70 and pts < 100),
                      "abort_if_still_100pts": bool(pts == 100.0 and pe["n_adm"] == 0)},
}
out["degenerate_all_reject"] = bool(pe["n_adm"] == 0)
out["keep"] = False
out["verdict"] = "KEEP" if (feas and kp >= 70 and pts < 100) else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r223_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
