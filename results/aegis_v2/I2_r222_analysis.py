"""Run 229: T222 = band-local median-only symmetric hard-keep control pooled-0 + 0% global,
single floor, zero extra filters (director iter 47, single-brain fallback
opencode-responses/muse-spark-1.3-contributor-free).

Frontier: discard-path bug, not tail statistic (upper-MAD / 2-tail IQR /
median-only + 0/20/50% anchor + EB on/off all = 100.0 discard).
Variation vs T221: strip IQR fence scaling AND C99 cap entirely:
  w_raw(e) = 1/(max(slip_m,FLOOR)+EPS) if t173(e) else 0.0   (single FLOOR, no C99 min-clip)
  s(e) = |w_raw(e) - med_b|                                  (raw, no fence, no scale,
                                                              no max(0,.), no soft, no EB/MAD)
Admit iff s <= TAU*_b (hard-keep central; per-band ROC, pooled-0: 8-pt
success-quantile grid Q{0.5..0.9}, recall_b>=0.65 then min pf, loosest tie).
Bands=CALIB jerk tertiles. Frozen I7+I10. Zero rig edits, frozen R173 files.
Why: T221 proved center-collapsed under fence scaling (s identically 0 ->
all 3 bands infeasible -> all-reject 100.0). T222 tests whether the collapse
is a scaling artifact (fence halfwidth swallows everything) rather than a
data artifact: raw |w-med| MUST have variance>0 and decile separation unless
w itself is degenerate.
Validation: same split; report keep%, s dist, NaN/Inf rate, discard counter.
PASS: keep>=5% breaks streak -> re-add anchor/pooling one at a time.
Predicted 65-75pts, keep 10-20%.
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
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
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
    # single floor, NO C99 min/max clip (director: raw)
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
cal_fails_band = [[e for e in cal_band[b] if not e["success"]] for b in range(3)]
cal_succ_band = [[e for e in cal_band[b] if e["success"]] for b in range(3)]

w_all = [wraw(e) for e in cal_eps]
MED_GLOB = median(w_all)
med_band = []
wvar_band = []
for b in range(3):
    ws = [wraw(e) for e in cal_band[b]]
    med_band.append(median(ws))
    mu = sum(ws) / len(ws)
    wvar_band.append(sum((x - mu) ** 2 for x in ws) / len(ws))


def mk_raw(centers):
    def s(e):
        return abs(wraw(e) - centers[band_of(e)])
    return s


s_main = mk_raw(med_band)


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


def fit_perband(score_fn, succ_bands):
    tau_star = {}
    for b in range(3):
        cand = sorted(score_fn(e) for e in succ_bands[b])
        if not cand:
            tau_star[b] = {"TAU": None, "feasible": False}
            continue
        vals = []
        for q in Q_GRID:
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


tau_main = fit_perband(s_main, cal_succ_band)
feas = all(tau_main[b]["feasible"] for b in range(3))
if feas:
    _t = {b: tau_main[b]["TAU"] for b in range(3)}
    A = lambda e, _t=_t: s_main(e) <= _t[band_of(e)]
else:
    A = lambda e: False


def audit_finite(fn, eps):
    bad = sum(1 for e in eps if not (math.isfinite(fn(e)) and fn(e) >= 0))
    return bad


nan_calib = audit_finite(s_main, cal_eps)
nan_held = audit_finite(s_main, tst_eps)
nan_arch = audit_finite(s_main, arc_eps)

# s distribution: pooled calib succ/fail deciles + per-band s stats
sc_s = sorted(s_main(e) for e in cal_eps if e["success"])
sc_f = sorted(s_main(e) for e in cal_eps if not e["success"])


def dec(x, q):
    return round(qceil(x, q), 4) if x else None


def var(xs):
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


sdist = {
    "succ_q10": dec(sc_s, 0.1), "succ_q50": dec(sc_s, 0.5), "succ_q90": dec(sc_s, 0.9),
    "fail_q10": dec(sc_f, 0.1), "fail_q50": dec(sc_f, 0.5), "fail_q90": dec(sc_f, 0.9),
    "succ_var": round(var(sc_s), 4) if sc_s else 0.0,
    "fail_var": round(var(sc_f), 4) if sc_f else 0.0,
    "pooled_var": round(var(sorted(s_main(e) for e in cal_eps)), 4),
}
sband = []
for b in range(3):
    vs = sorted(s_main(e) for e in cal_band[b])
    sband.append({"med": dec(vs, 0.5), "q90": dec(vs, 0.9), "var": round(var(vs), 4),
                  "frac_zero": round(sum(1 for v in vs if v == 0.0) / len(vs), 4)})

# discard counter: trailing run-keyed 100.0 discard streak + degenerate flag (filled post-eval)
trail_streak = 5  # measured pre-run: runs 224-228 metric==100.0 discard (run-keyed)

# LOBO
lobo_pts = {}
for left in range(3):
    pool = [e for b in range(3) for e in cal_band[b] if b != left]
    succ = sorted(s_main(e) for e in pool if e["success"])
    vs = []
    for q in Q_GRID:
        v = qceil(succ, q)
        if not vs or abs(v - vs[-1]) > 1e-12:
            vs.append(v)
    b2 = None
    for tv in vs:
        s2 = pooled_stats(lambda e, tv=tv: s_main(e) <= tv, pool)
        if s2["recall"] >= 0.65:
            key = (s2["pf"], -tv)
            if b2 is None or key < b2[0]:
                b2 = (key, tv, s2)
    if b2 is None:
        lobo_pts[str(left)] = {"feasible": False, "held_recall_pts": 0.0}
    else:
        tv = b2[1]
        ps = pooled_stats(lambda e, tv=tv: s_main(e) <= tv, tst_eps)
        lobo_pts[str(left)] = {"feasible": True, "held_recall_pts": round(100 * ps["recall"], 2),
                               "TAU": round(tv, 4)}
lobo_vals = [lobo_pts[str(b)]["held_recall_pts"] for b in range(3)]
lobo_var = round(max(lobo_vals) - min(lobo_vals), 2)

# Ablations (re-add one at a time candidates): anchor20/50 + pooled-8 single global TAU


def abl_perband(score_fn):
    succ = [[e for e in cal_band[b] if e["success"]] for b in range(3)]
    td = fit_perband(score_fn, succ)
    ok = all(td[b]["feasible"] for b in range(3))
    if not ok:
        fn = lambda e: False
    else:
        _t = {b: td[b]["TAU"] for b in range(3)}
        fn = lambda e, _t=_t: score_fn(e) <= _t[band_of(e)]
    ps = pooled_stats(fn, tst_eps)
    return {"feasible": ok, "held_recall_pts": round(100 * ps["recall"], 2), "n_adm": ps["n_adm"]}


abls = {}
for alabel, aa in [("anchor20", 0.2), ("anchor50", 0.5)]:
    c = [(1 - aa) * med_band[b] + aa * MED_GLOB for b in range(3)]
    abls[alabel] = abl_perband(mk_raw(c))
cand8 = sorted(s_main(e) for e in cal_eps if e["success"])
vs8 = []
for q in Q_GRID:
    v = qceil(cand8, q)
    if not vs8 or abs(v - vs8[-1]) > 1e-12:
        vs8.append(v)
b8 = None
for tv in vs8:
    s2 = pooled_stats(lambda e, tv=tv: s_main(e) <= tv, cal_eps)
    if s2["recall"] >= 0.65:
        key = (s2["pf"], -tv)
        if b8 is None or key < b8[0]:
            b8 = (key, tv, s2)
if b8 is None:
    abls["pooled8"] = {"feasible": False, "held_recall_pts": 100.0, "n_adm": 0}
else:
    tv8 = b8[1]
    fn8 = lambda e, tv8=tv8: s_main(e) <= tv8
    ps8 = pooled_stats(fn8, tst_eps)
    abls["pooled8"] = {"feasible": True, "held_recall_pts": round(100 * ps8["recall"], 2),
                       "n_adm": ps8["n_adm"], "TAU": round(tv8, 4)}

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


e222 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = pooled_eval(A)
pts = pe["score_recall_pts"]
kp = e222["keep_pct"]

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

out = {
    "variant": "T222 = band-local median-only symmetric hard-keep control pooled-0 + 0% global, single floor, zero extra: s=|wraw-med_b| raw (no C99 clip, no fence/scale/soft/EB/MAD); admit iff s<=TAU*_b (per-band 8-pt success grid)",
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [40, 40, 40],
              "cal_fails": [len(x) for x in cal_fails_band],
              "med_band": [round(v, 4) for v in med_band],
              "MED_global": round(MED_GLOB, 4),
              "anchor": 0.0,
              "wraw_var_band": [round(v, 4) for v in wvar_band],
              "wraw_var_gt0": bool(all(v > 0 for v in wvar_band)),
              "wraw_no_C99_cap": True},
    "fit": {"tau_star_per_band": {str(b): tau_main[b] for b in range(3)},
            "feasible": feas},
    "s_dist": dict(sdist, per_band=sband),
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
    "heldB": {"T222": e222, "cov_all": cov_all},
    "perband_held": band_held,
    "pooled_held": {"T222_pooled0": pe,
                    "recall_vs_T207_pts": round(pts - T207_REF, 2),
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["p_fail_adm"]), 2)},
    "archive_adm_frac_T222": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e222["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"predicted_pts_65_75": pts, "predicted_keep_10_20": kp,
                      "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0),
                      "keep_ge5_breaks_streak": streak_broken},
}
out["degenerate_all_reject"] = bool(pe["n_adm"] == 0)
out["keep"] = False
out["verdict"] = "KEEP-LINE" if (feas and streak_broken) else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r222_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
