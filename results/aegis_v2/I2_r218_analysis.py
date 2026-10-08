"""Run 225: T218 = band-local upper-tail EB-shrunk dual-floored hard-score pooled-0
+ 10% global anchor (director iter 43, single-brain fallback
opencode-responses/muse-spark-1.3-contributor-free).

Frontier: band-local robust upper-tail (T215/T217 100pts core), discard cause =
  pooled-0 + MAD_b~0 instability.
Variation T218: keep numerator/denom shrink + floor, add only light global anchor;
  no signed/hierarchical full pull like failed T216 53.3pts.
Formula: s = max(0, w191 - [0.9*med_b + 0.1*med_g])
            / (1.4826*max(MAD_b, 0.1*MAD_glob, floor) + eps),
  floor = (10*eps + 40*med_noise) / n_b^0.5.
Constant mapping (documented assumption — director names eps/med_noise without
  binding them to rig symbols; only two noise-floor constants exist in rig):
  eps = MAD_EPS = 1e-9; med_noise = EPS_slip = 0.005 (per-episode slip floor).
  => floor = (1e-8 + 0.2)/sqrt(40) ~= 0.0316 per band (n_b=40 all bands).
  Denominator uses RAW MAD_b (no EB shrinkage — dual floor replaces EB role).
Bands = CALIB jerk tertiles; per-band ROC TAU*_band (recall_b>=0.65 then min pf);
  pooled-0 eval with all-reject fallback if any band infeasible.
Frozen I7+I10. Zero rig edits, frozen R173 files. G7-clean, seg 15.
Validation: LOBO (leave-one-CALIB-band-out global-TAU refit, var<15pts) +
  MAD_b=0 synthetic slice + n_b<40 subsample slice; require var<15pts, no Inf/NaN.
Keep: predicted 85-95pts, KEEP iff pts>=70 AND heldB>=70 AND cov_ok AND finite.
Abort: pts < T217-5 (=95.0) or pooled-1 ablation wins.
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
MAD_SCALE = 1.4826
MAD_EPS = 1e-9
MEDNoise = EPS  # documented assumption: med_noise := slip floor 0.005
ANCHOR = 0.1  # 10% global anchor on median
GFLOOR_FRAC = 0.1  # max(MAD_b, 0.1*MAD_glob, floor)
T207_REF = 60.0
PLATEAU = 46.67
T217_PTS = 100.0


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


def spearman(xs, ys):
    n = len(xs)
    assert n == len(ys) and n > 1
    ox = sorted(range(n), key=lambda i: xs[i])
    oy = sorted(range(n), key=lambda i: ys[i])
    rxi = [0] * n
    ryi = [0] * n
    for r, i in enumerate(ox):
        rxi[i] = r
    for r, i in enumerate(oy):
        ryi[i] = r
    mx = sum(rxi) / n
    my = sum(ryi) / n
    cov = sum((rxi[i] - mx) * (ryi[i] - my) for i in range(n))
    vx = sum((r - mx) ** 2 for r in rxi)
    vy = sum((r - my) ** 2 for r in ryi)
    if vx == 0 or vy == 0:
        return 1.0 if xs == ys else 0.0
    return cov / math.sqrt(vx * vy)


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

# T218 centers (10% global anchor) + dual-floored scales (raw MAD_b)
center_star = [(1 - ANCHOR) * med_band[b] + ANCHOR * MED_GLOB for b in range(3)]
floor_b = [(10 * MAD_EPS + 40 * MEDNoise) / math.sqrt(n_band[b]) for b in range(3)]
gfloor = GFLOOR_FRAC * MAD_GLOB
scale_star, floor_binds, gfloor_binds = [], [], []
for b in range(3):
    ms = max(mad_band[b], gfloor, floor_b[b])
    floor_binds.append(mad_band[b] < floor_b[b])
    gfloor_binds.append(mad_band[b] < gfloor and gfloor >= floor_b[b])
    sc = MAD_SCALE * ms + MAD_EPS
    if not (sc > 1e-12):
        sc = 1.0
    scale_star.append(sc)
floor_frac = sum(floor_binds) / 3.0
collapse = [md < 1e-9 for md in mad_band]
collapse_frac = sum(collapse) / 3.0

# T217 ref scales for rank correlation (recomputed identically)
_, _, _, _ = None, None, None, None
t217_scales = []
for b in range(3):
    eb = (n_band[b] * mad_band[b] + 10 * MAD_GLOB) / (n_band[b] + 10)
    ms = max(eb, 0.15 * MAD_GLOB, MAD_EPS)
    t217_scales.append(MAD_SCALE * ms)
t217_med = list(med_band)


def mk_s(scales, centers):
    def s(e):
        b = band_of(e)
        return max(0.0, w191(e) - centers[b]) / scales[b]
    return s


s_main = mk_s(scale_star, center_star)
s_t217 = mk_s(t217_scales, t217_med)


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


tau_main = fit_perband(s_main)
feas_main = all(tau_main[b]["feasible"] for b in range(3))
TAUS = {b: tau_main[b]["TAU"] for b in range(3)} if feas_main else None


def mk_admit(tau_d, score_fn):
    if not all(tau_d[b]["feasible"] for b in range(3)):
        return (lambda e: False), False
    _t = {b: tau_d[b]["TAU"] for b in range(3)}
    return (lambda e, _t=_t: score_fn(e) >= _t[band_of(e)]), True


A, feas = (mk_admit(tau_main, s_main) if feas_main else ((lambda e: False), False))


def audit_finite(score_fn, eps):
    bad = 0
    for e in eps:
        v = score_fn(e)
        if not (math.isfinite(v) and v >= 0):
            bad += 1
    return bad


nan_calib = audit_finite(s_main, cal_eps)
nan_held = audit_finite(s_main, tst_eps)
nan_arch = audit_finite(s_main, arc_eps)

# --- Stress slice 1: synthetic MAD_b=0 per band (floor must cap blowup) ---
mad0_demo = {}
for b in range(3):
    ms = max(0.0, gfloor, floor_b[b])
    sc = MAD_SCALE * ms + MAD_EPS
    mad0_demo[str(b)] = {"MAD_b_set0": True, "scale": round(sc, 4),
                         "score_per_10units": round(10.0 / sc, 4),
                         "finite": bool(math.isfinite(10.0 / sc))}

# --- Stress slice 2: n_b<40 subsample refit (first-n deterministic, n in 10/20) ---
sub_results = {}
for n in (10, 20):
    sub = [cal_band[b][:n] for b in range(3)]
    sub_med, sub_mad = [], []
    for b in range(3):
        ws = [w191(e) for e in sub[b]]
        m = median(ws)
        sub_med.append(m)
        sub_mad.append(mad(ws, m))
    sub_c = [(1 - ANCHOR) * sub_med[b] + ANCHOR * MED_GLOB for b in range(3)]
    sub_sc = []
    for b in range(3):
        fl = (10 * MAD_EPS + 40 * MEDNoise) / math.sqrt(n)
        ms = max(sub_mad[b], gfloor, fl)
        sub_sc.append(MAD_SCALE * ms + MAD_EPS)
    s_sub = mk_s(sub_sc, sub_c)
    sub_fails = [[e for e in sub[b] if not e["success"]] for b in range(3)]
    t_sub = {}
    ok = True
    for b in range(3):
        if not sub_fails[b]:
            t_sub[b] = {"TAU": None, "feasible": False}
            ok = False
            continue
        vals = []
        for q in Q_GRID:
            v = qceil(sorted(s_sub(e) for e in sub_fails[b]), q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        best = None
        for tv in vals:
            s = inband_stats(lambda e, tv=tv: s_sub(e) >= tv, sub[b])
            if s["recall"] >= 0.65:
                key = (s["pf"], tv)
                if best is None or key < best[0]:
                    best = (key, tv, s)
        t_sub[b] = {"TAU": best[1] if best else None, "feasible": best is not None}
        ok = ok and best is not None
    if ok:
        _t = {b: t_sub[b]["TAU"] for b in range(3)}
        Asub = lambda e, _t=_t: s_sub(e) >= _t[band_of(e)]
    else:
        Asub = lambda e: False
    ps = pooled_stats(Asub, tst_eps)
    bad = audit_finite(s_sub, tst_eps)
    sub_results[str(n)] = {"feasible": ok, "pooled_recall_pts": round(100 * ps["recall"], 2),
                           "n_adm": ps["n_adm"], "nonfinite_held": bad}

# --- LOBO: leave-one-CALIB-band-out global-TAU refit -> held pooled recall ---
lobo_pts = {}
for left in range(3):
    pool = [e for b in range(3) for e in cal_band[b] if b != left]
    fails = sorted(s_main(e) for e in pool if not e["success"])
    vals = []
    for q in Q_GRID:
        v = qceil(fails, q)
        if not vals or abs(v - vals[-1]) > 1e-12:
            vals.append(v)
    best = None
    for tv in vals:
        s = pooled_stats(lambda e, tv=tv: s_main(e) >= tv, pool)
        if s["recall"] >= 0.65:
            key = (s["pf"], tv)
            if best is None or key < best[0]:
                best = (key, tv, s)
    if best is None:
        lobo_pts[str(left)] = {"feasible": False, "held_recall_pts": 0.0}
    else:
        tv = best[1]
        ps = pooled_stats(lambda e, tv=tv: s_main(e) >= tv, tst_eps)
        lobo_pts[str(left)] = {"feasible": True, "held_recall_pts": round(100 * ps["recall"], 2),
                               "TAU": round(tv, 4)}
lobo_vals = [lobo_pts[str(b)]["held_recall_pts"] for b in range(3)]
lobo_var = round(max(lobo_vals) - min(lobo_vals), 2)

# --- pooled-1 ablation (abort leg: does pooled-1 win?) ---
fails_all = sorted(s_main(e) for e in cal_eps if not e["success"])
vals = []
for q in Q_GRID:
    v = qceil(fails_all, q)
    if not vals or abs(v - vals[-1]) > 1e-12:
        vals.append(v)
best = None
for tv in vals:
    s = pooled_stats(lambda e, tv=tv: s_main(e) >= tv, cal_eps)
    if s["recall"] >= 0.65:
        key = (s["pf"], tv)
        if best is None or key < best[0]:
            best = (key, tv, s)
if best is None:
    p1 = {"feasible": False, "held_recall_pts": 0.0, "TAU": None}
    A1 = lambda e: False
else:
    tv1 = best[1]
    ps1 = pooled_stats(lambda e, tv=tv1: s_main(e) >= tv1, tst_eps)
    p1 = {"feasible": True, "held_recall_pts": round(100 * ps1["recall"], 2),
          "TAU": round(tv1, 4), "n_adm": ps1["n_adm"]}
    A1 = lambda e, tv1=tv1: s_main(e) >= tv1

held_s_main = [s_main(e) for e in tst_eps]
held_s_t217 = [s_t217(e) for e in tst_eps]
rank_corr = round(spearman(held_s_main, held_s_t217), 4)

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


e218 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe_main = pooled_eval(A)
pe_p1 = pooled_eval(A1)
pts = pe_main["score_recall_pts"]
kp = e218["keep_pct"]

band_held = {}
for b in range(3):
    hb = [e for e in tst_eps if band_of(e) == b]
    hf = [e for e in hb if not e["success"]]
    ha = [e for e in hb if A(e)]
    prec_b = (sum(1 for e in ha if e["success"]) / len(ha)) if ha else 1.0
    rec_b = (sum(1 for e in hf if not A(e)) / len(hf)) if hf else 1.0
    band_held[b] = {"n": len(hb), "n_fails": len(hf), "n_adm": len(ha),
                    "recall_band": round(rec_b, 4), "precision_band": round(prec_b, 4)}

sub_pts = [sub_results[str(n)]["pooled_recall_pts"] for n in (10, 20)] + [pts]
stress_var = round(max(sub_pts) - min(sub_pts), 2)

out = {
    "variant": "T218 = band-local upper-tail dual-floored hard-score pooled-0 + 10% global anchor: s=max(0,w191-[0.9*med_b+0.1*med_g])/(1.4826*max(MAD_b,0.1*MAD_glob,floor)+eps), floor=(10*eps+40*med_noise)/sqrt(n_b), eps=1e-9, med_noise=0.005",
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [len(x) for x in cal_band],
              "cal_fails": [len(x) for x in cal_fails_band],
              "med_band": [round(v, 4) for v in med_band],
              "MAD_band": [round(v, 4) for v in mad_band],
              "MAD_global": round(MAD_GLOB, 4),
              "MED_global": round(MED_GLOB, 4),
              "n_band": n_band,
              "anchor": ANCHOR,
              "center_star": [round(v, 4) for v in center_star],
              "gfloor_01": round(gfloor, 4),
              "floor_b": [round(v, 4) for v in floor_b],
              "scale_star": [round(v, 4) for v in scale_star],
              "floor_binds": floor_binds, "floor_frac": round(floor_frac, 4),
              "gfloor_binds": gfloor_binds,
              "collapse_raw_MAD": collapse, "collapse_frac": round(collapse_frac, 4)},
    "fit": {"tau_star_per_band": {str(b): tau_main[b] for b in range(3)},
            "TAU_STAR_band": ({str(b): round(TAUS[b], 4) for b in range(3)} if TAUS else None),
            "feasible": feas},
    "finite_audit": {"nonfinite_or_neg_calib": nan_calib, "nonfinite_or_neg_held": nan_held,
                     "nonfinite_or_neg_arch": nan_arch,
                     "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0)},
    "mad0_slice": mad0_demo,
    "nsub_slice": sub_results,
    "stress_var_subsample_pts": stress_var,
    "lobo": {"per_leftout": lobo_pts, "var_pts": lobo_var, "pass_var_lt15": bool(lobo_var < 15)},
    "pooled1_ablation": p1,
    "pooled1_wins": bool(p1["held_recall_pts"] > pts),
    "abort_below_T217m5": bool(pts < T217_PTS - 5),
    "rank_corr_vs_T217_held_spearman": rank_corr,
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T218": e218, "cov_all": cov_all},
    "perband_held": band_held,
    "pooled_held": {"T218_pooled0": pe_main,
                    "recall_vs_T207_pts": round(pts - T207_REF, 2),
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe_main["p_fail_adm"]), 2)},
    "archive_adm_frac_T218": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e218["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"pts_ge_70": bool(pts >= 70), "heldB_keep_ge_70": bool(kp >= 70),
                      "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0),
                      "lobo_var_lt15": bool(lobo_var < 15),
                      "predicted_pts_85_95": pts, "predicted_keep": kp},
}
discard = (not feas or pts < 70 or kp < 70 or (pe_main["n_adm"] == 0)
           or not out["cov_ok"]
           or not (nan_calib == 0 and nan_held == 0 and nan_arch == 0))
out["degenerate_all_reject"] = bool(pe_main["n_adm"] == 0)
out["verdict"] = "DISCARD" if discard else "KEEP"
out["keep"] = bool(out["verdict"] == "KEEP" and out["runA_keep_cited"])
json.dump(out, open("results/aegis_v2/I2_r218_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
