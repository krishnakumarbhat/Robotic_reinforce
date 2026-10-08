"""Run 231: T224 = band-local median asymmetric upper-only IQR soft-clip pooled-2
+ 10% global anchor + light EB (director iter 49, single-brain fallback
opencode-responses/muse-spark-1.3-contributor-free).

Frontier: keep-first. T221-T223 all DISCARD on keep (0.0), not score (100.0
degenerate artifact). T223 (pooled-4 grid + 25% anchor) over-shrinks: thr far
above med -> frac_zero 0.75-1.0. T221/T222 (pooled-0/8-pt, 0% anchor) too
strict: center-collapse / central-keep infeasible. T224 interpolates.
Variation vs T223 (keeps s=max(0,wraw-thr_b) + single lower floor at med_b,
zero-floor off, still no MAD extra filter):
  wraw(e) = 1/(max(slip_m,FLOOR)+EPS) if t173(e) else 0.0 (single floor, no C99)
  c_b     = 0.90*med_b + 0.10*med_g          (10% global anchor; was 25%)
  IQRup*_b = (n_b*IQRup_b + 2*IQRup_g)/(n_b+2) (light EB k=2 only; no MAD filter)
  thr_b   = max(c_b + 1.5*IQRup*_b, med_b)    (single lower floor, zero-floor off)
  s(e)    = max(0, wraw(e) - thr_b)
  p(e)    = 1 - exp(-s / max(IQRup*_b, 1e-9))  (soft-clip, monotonic)
Admit iff p <= TAU*_b (per-band ROC, pooled-2 = 2-pt success-quantile grid
Q{0.65,0.8}, recall_b>=0.65 then min pf, loosest tie).
Bands=CALIB jerk tertiles. Frozen I7+I10. Zero rig edits, frozen R173 files.
Why: pooled-0 (8-pt) too strict, pooled-4 + 25% over-shrinks; 2 + 10% lands
between: thr drops vs T223 (anchor 0.25->0.10 pulls c_b down; EB k=2 barely
moves IQRup at n_b=40) while grid coarsens (fewer strict cutpoints).
Validation: same 228-230 split/scoring + ablation pooled-grid {8,2,4} x global
{0,10,25}%-anchor (9 cells, EB k=2 fixed) + bonus pooled-2 x 15% cell answering
the director stop-rule without a new iteration. Report keep/pts per cell.
Predicted 92-96pts with keep 72-78 -> new BEST. Verdict KEEP iff keep>=70
(discard on keep, not score). Stop-rule: if main keep<70 -> next is T225 with
global 15%, band logic untouched.
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
Q_GRID8 = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
Q_GRID4 = [0.5, 0.65, 0.8, 0.9]
Q_GRID2 = [0.65, 0.8]
ANCHOR_MAIN = 0.10
EB_K = 2
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

w_all = sorted(wraw(e) for e in cal_eps)
MED_GLOB = median(w_all)
IQRUP_GLOB = max(qceil(w_all, 0.75) - MED_GLOB, 0.0)

med_band, iqr_raw = [], []
for b in range(3):
    ws = sorted(wraw(e) for e in cal_band[b])
    mb = median(ws)
    med_band.append(mb)
    iqr_raw.append(max(qceil(ws, 0.75) - mb, 0.0))

N_B = 40
iqr_eb = [(N_B * iqr_raw[b] + EB_K * IQRUP_GLOB) / (N_B + EB_K) for b in range(3)]
scale_b = [max(v, 1e-9) for v in iqr_eb]


def build(anchor):
    c = [(1 - anchor) * med_band[b] + anchor * MED_GLOB for b in range(3)]
    thr = [max(c[b] + 1.5 * iqr_eb[b], med_band[b]) for b in range(3)]

    def p_fn(e, _c=c, _thr=thr):
        b = band_of(e)
        s = max(0.0, wraw(e) - _thr[b])
        return 1.0 - math.exp(-s / scale_b[b])

    def s_fn(e, _thr=thr):
        return max(0.0, wraw(e) - _thr[band_of(e)])

    return c, thr, p_fn, s_fn


C_MAIN, THR_MAIN, P_MAIN, S_MAIN = build(ANCHOR_MAIN)


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


tau_main = fit_perband(P_MAIN, cal_succ_band, Q_GRID2)
feas = all(tau_main[b]["feasible"] for b in range(3))
if feas:
    _t = {b: tau_main[b]["TAU"] for b in range(3)}
    A = lambda e, _t=_t: P_MAIN(e) <= _t[band_of(e)]
else:
    A = lambda e: False


def audit_finite(fn, eps):
    return sum(1 for e in eps if not (math.isfinite(fn(e)) and 0.0 <= fn(e) <= 1.0))


nan_calib = audit_finite(P_MAIN, cal_eps)
nan_held = audit_finite(P_MAIN, tst_eps)
nan_arch = audit_finite(P_MAIN, arc_eps)

sc_s = sorted(P_MAIN(e) for e in cal_eps if e["success"])
sc_f = sorted(P_MAIN(e) for e in cal_eps if not e["success"])


def dec(x, q):
    return round(qceil(x, q), 4) if x else None


def var(xs):
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


sdist = {
    "succ_q10": dec(sc_s, 0.1), "succ_q50": dec(sc_s, 0.5), "succ_q90": dec(sc_s, 0.9),
    "fail_q10": dec(sc_f, 0.1), "fail_q50": dec(sc_f, 0.5), "fail_q90": dec(sc_f, 0.9),
    "pooled_var": round(var(sorted(P_MAIN(e) for e in cal_eps)), 6),
}
sband = []
for b in range(3):
    vs = sorted(P_MAIN(e) for e in cal_band[b])
    sband.append({"med": dec(vs, 0.5), "q90": dec(vs, 0.9), "var": round(var(vs), 6),
                  "frac_zero": round(sum(1 for v in vs if v == 0.0) / len(vs), 4)})

trail_streak = 7

lobo_pts = {}
for left in range(3):
    pool = [e for b in range(3) for e in cal_band[b] if b != left]
    succ = sorted(P_MAIN(e) for e in pool if e["success"])
    vs = []
    for q in Q_GRID2:
        v = qceil(succ, q)
        if not vs or abs(v - vs[-1]) > 1e-12:
            vs.append(v)
    b2 = None
    for tv in vs:
        s2 = pooled_stats(lambda e, tv=tv: P_MAIN(e) <= tv, pool)
        if s2["recall"] >= 0.65:
            key = (s2["pf"], -tv)
            if b2 is None or key < b2[0]:
                b2 = (key, tv, s2)
    if b2 is None:
        lobo_pts[str(left)] = {"feasible": False, "held_recall_pts": 0.0}
    else:
        tv = b2[1]
        ps = pooled_stats(lambda e, tv=tv: P_MAIN(e) <= tv, tst_eps)
        lobo_pts[str(left)] = {"feasible": True, "held_recall_pts": round(100 * ps["recall"], 2),
                               "TAU": round(tv, 4)}
lobo_vals = [lobo_pts[str(b)]["held_recall_pts"] for b in range(3)]
lobo_var = round(max(lobo_vals) - min(lobo_vals), 2)


def cell_eval(grid, anchor):
    _, _, p_fn, _ = build(anchor)
    succ = [[e for e in cal_band[b] if e["success"]] for b in range(3)]
    td = fit_perband(p_fn, succ, grid)
    ok = all(td[b]["feasible"] for b in range(3))
    if not ok:
        fn = lambda e: False
    else:
        _t = {b: td[b]["TAU"] for b in range(3)}
        fn = lambda e, _t=_t: p_fn(e) <= _t[band_of(e)]
    ps = pooled_stats(fn, tst_eps)
    hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
    admB = [e for e in hB if fn(e)]
    keep = round(100 * sum(1 for e in admB if e["success"]) / 20.0, 2)
    return {"feasible": ok, "held_recall_pts": round(100 * ps["recall"], 2),
            "n_adm": ps["n_adm"], "heldB_keep_pct": keep,
            "taus": [td[b]["TAU"] for b in range(3)]}


GRIDS = {"pooled-0": Q_GRID8, "pooled-2": Q_GRID2, "pooled-4": Q_GRID4}
abls = {}
for gname, grid in GRIDS.items():
    for aname, anchor in [("global-0", 0.0), ("global-10", 0.10), ("global-25", 0.25)]:
        abls["%s_x_%s" % (gname, aname)] = cell_eval(grid, anchor)
abls["pooled-2_x_global-15_bonus"] = cell_eval(Q_GRID2, 0.15)

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


e224 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = pooled_eval(A)
pts = pe["score_recall_pts"]
kp = e224["keep_pct"]

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

out = {
    "variant": "T224 = band-local median asymmetric upper-only IQR soft-clip pooled-2 + 10% global anchor + light EB k=2: s=max(0,wraw-thr_b), thr_b=max(0.9*med_b+0.1*med_g+1.5*IQRup*_b,med_b), IQRup*_b=(40*IQRup_b+2*IQRup_g)/42, p=1-exp(-s/max(IQRup*_b,eps)); admit iff p<=TAU*_b (per-band 2-pt success grid Q{0.65,0.8})",
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [40, 40, 40],
              "med_band": [round(v, 4) for v in med_band],
              "MED_global": round(MED_GLOB, 4),
              "anchor": ANCHOR_MAIN,
              "IQRup_raw": [round(v, 4) for v in iqr_raw],
              "IQRup_glob": round(IQRUP_GLOB, 4),
              "IQRup_EB_k2": [round(v, 4) for v in iqr_eb],
              "EB_shrink_delta": [round(iqr_eb[b] - iqr_raw[b], 4) for b in range(3)],
              "center_star": [round(C_MAIN[b], 4) for b in range(3)],
              "thr_band": [round(v, 4) for v in THR_MAIN],
              "thr_vs_T223": "lower (anchor 0.10<0.25; EB k=2 near-identity at n=40)",
              "floor_at_med_binds": [bool(abs(THR_MAIN[b] - med_band[b]) < 1e-12) for b in range(3)],
              "wraw_no_C99_cap": True,
              "symmetric_off": True, "MAD_extra_filter_off": True, "EB_k": EB_K},
    "fit": {"tau_star_per_band": {str(b): tau_main[b] for b in range(3)},
            "feasible": feas, "grid": "Q2 pooled-2 {0.65,0.8}"},
    "s_dist": dict(sdist, per_band=sband),
    "finite_audit": {"nonfinite_or_neg_calib": nan_calib,
                     "nonfinite_or_neg_held": nan_held,
                     "nonfinite_or_neg_arch": nan_arch,
                     "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0)},
    "discard_counter": {"trailing_100_streak_pre": trail_streak,
                        "degenerate_now": bool(pe["n_adm"] == 0),
                        "counter": discard_counter},
    "lobo": {"per_leftout": lobo_pts, "var_pts": lobo_var,
             "pass_var_lt15": bool(lobo_var < 15)},
    "ablations_pooled_x_global_EB_k2_fixed": abls,
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T224": e224, "cov_all": cov_all},
    "perband_held": band_held,
    "pooled_held": {"T224_pooled2": pe,
                    "recall_vs_T207_pts": round(pts - T207_REF, 2),
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["p_fail_adm"]), 2)},
    "archive_adm_frac_T224": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e224["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"predicted_pts_92_96": pts, "predicted_keep_72_78": kp,
                      "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0),
                      "success_keep_ge70": bool(kp >= 70),
                      "stop_rule_15pct_next": bool(kp < 70)},
}
out["degenerate_all_reject"] = bool(pe["n_adm"] == 0)
out["keep"] = bool(feas and kp >= 70)
out["verdict"] = "KEEP" if (feas and kp >= 70) else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r224_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
