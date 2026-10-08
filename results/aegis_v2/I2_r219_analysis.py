"""Run 226: T219 = band-local upper-tail asymmetric upper-MAD EB-lite single-floor
soft-score pooled-0 + 20% global anchor (director iter 44, single-brain fallback
opencode-responses/muse-spark-1.3-contributor-free).

Frontier: asymmetric scale. T216-218 all use symmetric MAD_b, fatal if lower-tail
tight/zero while upper-tail loose.
Variation vs T217/T218: replace 40*MAD symmetric + dual-floor + hard-score with
1x upper-MAD + single floor + soft p=1-exp(-s).
Formula:
  med*_b = 0.8*med_b + 0.2*med_global
  MADup_b = median(|x - med_b| for x >= med_b)   (upper-tail only)
  MADup_global = median(|x - med_g| for x >= med_g) over all CALIB
  w = n_b/(n_b+20);  MADup*_b = w*MADup_b + (1-w)*MADup_global
  floor = max(5, 0.5*MADup_global)               (single floor)
  s = max(0, w191 - med*_b) / (1.4826*max(MADup*_b, floor))
  p = 1 - exp(-s)                                (soft score, monotonic in s)
Admit iff p >= TAU_p,b (per-band ROC on CALIB pooled-0, recall_b>=0.65 then min
pf_b); all-reject fallback if any band infeasible.
Bands = CALIB jerk tertiles. Frozen I7+I10. Zero rig edits, frozen R173 files.
G7-clean, seg 15.
Validation: zero-MAD_b synthetic slice, n_b<20 shrinkage-weight slice (n=10/20),
skew (MADup vs MADlow) + symmetric-MAD ablation, anchor 0/0.2 ablation, soft-vs-
hard rank stability (Spearman s vs p + TAU-bit-identical check), LOBO var<15,
pooled-1 ablation. Keep iff pts>=70 AND heldB>=70 AND cov_ok AND finite.
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
ANCHOR = 0.2  # 20% global anchor on median
SHRINK_K = 20  # w = n_b/(n_b+20)
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


def mad(xs, med):
    return st.median([abs(x - med) for x in xs])


def mad_up(xs, med):
    up = [x - med for x in xs if x >= med]
    if not up:
        return 0.0
    return st.median(up)


def mad_low(xs, med):
    lo = [med - x for x in xs if x <= med]
    if not lo:
        return 0.0
    return st.median(lo)


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

med_band, madup_band, madlow_band, madsym_band = [], [], [], []
for b in range(3):
    ws = [w191(e) for e in cal_band[b]]
    m = median(ws)
    med_band.append(m)
    madup_band.append(mad_up(ws, m))
    madlow_band.append(mad_low(ws, m))
    madsym_band.append(mad(ws, m))

w_all = [w191(e) for e in cal_eps]
MED_GLOB = median(w_all)
MADUP_GLOB = mad_up(w_all, MED_GLOB)
MADLOW_GLOB = mad_low(w_all, MED_GLOB)
MAD_GLOB = mad(w_all, MED_GLOB)
n_band = [len(cal_band[b]) for b in range(3)]

# T219 centers (20% global anchor) + EB-lite upper scales + single floor
wgt = [n_band[b] / (n_band[b] + SHRINK_K) for b in range(3)]
center_star = [(1 - ANCHOR) * med_band[b] + ANCHOR * MED_GLOB for b in range(3)]
madup_star = [wgt[b] * madup_band[b] + (1 - wgt[b]) * MADUP_GLOB for b in range(3)]
FLOOR1 = max(5.0, 0.5 * MADUP_GLOB)
scale_star = [MAD_SCALE * max(madup_star[b], FLOOR1) for b in range(3)]
floor_binds = [madup_star[b] < FLOOR1 for b in range(3)]
collapse = [md < 1e-9 for md in madup_band]
skew_ratio = [(madup_band[b] / madlow_band[b] if madlow_band[b] > 1e-12 else float("inf"))
              for b in range(3)]


def mk_s(scales, centers):
    def s(e):
        b = band_of(e)
        return max(0.0, w191(e) - centers[b]) / scales[b]
    return s


def mk_p(scales, centers):
    s = mk_s(scales, centers)
    return lambda e: 1.0 - math.exp(-s(e))


s_main = mk_s(scale_star, center_star)
p_main = mk_p(scale_star, center_star)
# symmetric-MAD ablation (same centers/floor-form, symmetric scale, EB-lite equally)
madsym_star = [wgt[b] * madsym_band[b] + (1 - wgt[b]) * MAD_GLOB for b in range(3)]
floor_sym = max(5.0, 0.5 * MAD_GLOB)
scale_sym = [MAD_SCALE * max(madsym_star[b], floor_sym) for b in range(3)]
p_sym = mk_p(scale_sym, center_star)
# anchor-0 ablation (pure local median, same upper scales)
center_a0 = list(med_band)
p_a0 = mk_p(scale_star, center_a0)


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


tau_main = fit_perband(p_main)
feas_main = all(tau_main[b]["feasible"] for b in range(3))
TAUS = {b: tau_main[b]["TAU"] for b in range(3)} if feas_main else None


def mk_admit(tau_d, score_fn):
    if not all(tau_d[b]["feasible"] for b in range(3)):
        return (lambda e: False), False
    _t = {b: tau_d[b]["TAU"] for b in range(3)}
    return (lambda e, _t=_t: score_fn(e) >= _t[band_of(e)]), True


A, feas = (mk_admit(tau_main, p_main) if feas_main else ((lambda e: False), False))
# hard-score (s) fit for soft-vs-hard stability check
tau_hard = fit_perband(s_main)
feas_hard = all(tau_hard[b]["feasible"] for b in range(3))
Ah, _ = (mk_admit(tau_hard, s_main) if feas_hard else ((lambda e: False), False))


def audit_finite(score_fn, eps):
    bad = 0
    for e in eps:
        v = score_fn(e)
        if not (math.isfinite(v) and v >= 0):
            bad += 1
    return bad


nan_calib = audit_finite(p_main, cal_eps)
nan_held = audit_finite(p_main, tst_eps)
nan_arch = audit_finite(p_main, arc_eps)

# --- Stress 1: synthetic zero upper-MAD per band (floor must cap) ---
mad0_demo = {}
for b in range(3):
    sc = MAD_SCALE * max(0.0 * wgt[b] + (1 - wgt[b]) * MADUP_GLOB, FLOOR1)
    mad0_demo[str(b)] = {"MADup_b_set0": True, "scale": round(sc, 4),
                         "s_per_10units": round(10.0 / sc, 4),
                         "p_per_10units": round(1 - math.exp(-10.0 / sc), 4),
                         "finite": bool(math.isfinite(10.0 / sc))}

# --- Stress 2: n_b<20 shrinkage-weight slice (first-n deterministic) ---
sub_results = {}
for n in (10, 20):
    sub = [cal_band[b][:n] for b in range(3)]
    sub_med, sub_up = [], []
    for b in range(3):
        ws = [w191(e) for e in sub[b]]
        m = median(ws)
        sub_med.append(m)
        sub_up.append(mad_up(ws, m))
    ww = [n / (n + SHRINK_K)] * 3
    sub_c = [(1 - ANCHOR) * sub_med[b] + ANCHOR * MED_GLOB for b in range(3)]
    sub_star = [ww[b] * sub_up[b] + (1 - ww[b]) * MADUP_GLOB for b in range(3)]
    sub_sc = [MAD_SCALE * max(sub_star[b], FLOOR1) for b in range(3)]
    p_sub = mk_p(sub_sc, sub_c)
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
            v = qceil(sorted(p_sub(e) for e in sub_fails[b]), q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        best = None
        for tv in vals:
            s = inband_stats(lambda e, tv=tv: p_sub(e) >= tv, sub[b])
            if s["recall"] >= 0.65:
                key = (s["pf"], tv)
                if best is None or key < best[0]:
                    best = (key, tv, s)
        t_sub[b] = {"TAU": best[1] if best else None, "feasible": best is not None}
        ok = ok and best is not None
    if ok:
        _t = {b: t_sub[b]["TAU"] for b in range(3)}
        Asub = lambda e, _t=_t: p_sub(e) >= _t[band_of(e)]
    else:
        Asub = lambda e: False
    ps = pooled_stats(Asub, tst_eps)
    bad = audit_finite(p_sub, tst_eps)
    sub_results[str(n)] = {"feasible": ok, "shrink_w": round(n / (n + SHRINK_K), 4),
                           "pooled_recall_pts": round(100 * ps["recall"], 2),
                           "n_adm": ps["n_adm"], "nonfinite_held": bad}

# --- LOBO: leave-one-CALIB-band-out global-TAU refit ---
lobo_pts = {}
for left in range(3):
    pool = [e for b in range(3) for e in cal_band[b] if b != left]
    fails = sorted(p_main(e) for e in pool if not e["success"])
    vals = []
    for q in Q_GRID:
        v = qceil(fails, q)
        if not vals or abs(v - vals[-1]) > 1e-12:
            vals.append(v)
    best = None
    for tv in vals:
        s = pooled_stats(lambda e, tv=tv: p_main(e) >= tv, pool)
        if s["recall"] >= 0.65:
            key = (s["pf"], tv)
            if best is None or key < best[0]:
                best = (key, tv, s)
    if best is None:
        lobo_pts[str(left)] = {"feasible": False, "held_recall_pts": 0.0}
    else:
        tv = best[1]
        ps = pooled_stats(lambda e, tv=tv: p_main(e) >= tv, tst_eps)
        lobo_pts[str(left)] = {"feasible": True, "held_recall_pts": round(100 * ps["recall"], 2),
                               "TAU": round(tv, 4)}
lobo_vals = [lobo_pts[str(b)]["held_recall_pts"] for b in range(3)]
lobo_var = round(max(lobo_vals) - min(lobo_vals), 2)

# --- pooled-1 ablation ---
fails_all = sorted(p_main(e) for e in cal_eps if not e["success"])
vals = []
for q in Q_GRID:
    v = qceil(fails_all, q)
    if not vals or abs(v - vals[-1]) > 1e-12:
        vals.append(v)
best = None
for tv in vals:
    s = pooled_stats(lambda e, tv=tv: p_main(e) >= tv, cal_eps)
    if s["recall"] >= 0.65:
        key = (s["pf"], tv)
        if best is None or key < best[0]:
            best = (key, tv, s)
if best is None:
    p1 = {"feasible": False, "held_recall_pts": 0.0, "TAU": None}
    A1 = lambda e: False
else:
    tv1 = best[1]
    ps1 = pooled_stats(lambda e, tv=tv1: p_main(e) >= tv1, tst_eps)
    p1 = {"feasible": True, "held_recall_pts": round(100 * ps1["recall"], 2),
          "TAU": round(tv1, 4), "n_adm": ps1["n_adm"]}
    A1 = lambda e, tv1=tv1: p_main(e) >= tv1

# --- anchor 0 / symmetric ablations (same pooled-0 protocol) ---
def abl_eval(score_fn):
    t = fit_perband(score_fn)
    ok = all(t[b]["feasible"] for b in range(3))
    fn, _ = (mk_admit(t, score_fn) if ok else ((lambda e: False), False))
    ps = pooled_stats(fn, tst_eps)
    return {"feasible": ok, "held_recall_pts": round(100 * ps["recall"], 2),
            "n_adm": ps["n_adm"],
            "TAU": ({str(b): round(t[b]["TAU"], 4) for b in range(3)} if ok else None)}


abl_anchor0 = abl_eval(p_a0)
abl_sym = abl_eval(p_sym)

held_s = [s_main(e) for e in tst_eps]
held_p = [p_main(e) for e in tst_eps]
rank_soft_hard = round(spearman(held_s, held_p), 4)
# hard-vs-soft admit agreement on held
agree = sum(1 for e in tst_eps if A(e) == Ah(e)) / len(tst_eps)

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
    pf = sum(1 for e in adm if e["success"]) / nn if nn else 0.0
    prec = sum(1 for e in adm if e["success"]) / nn if nn else 1.0
    return {"n_adm": nn, "veto_rate": round(1 - nn / len(tst_eps), 4),
            "p_fail_adm": round(pf, 4),
            "precision_adm": round(prec, 4),
            "recall_fail_reject": round(caught / len(fails_h), 4),
            "score_recall_pts": round(100 * caught / len(fails_h), 2)}


e219 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe_main = pooled_eval(A)
pe_p1 = pooled_eval(A1)
pts = pe_main["score_recall_pts"]
kp = e219["keep_pct"]

band_held = {}
for b in range(3):
    hb = [e for e in tst_eps if band_of(e) == b]
    hf = [e for e in hb if not e["success"]]
    ha = [e for e in hb if A(e)]
    prec_b = (sum(1 for e in ha if e["success"]) / len(ha)) if ha else 1.0
    rec_b = (sum(1 for e in hf if not A(e)) / len(hf)) if hf else 1.0
    band_held[b] = {"n": len(hb), "n_fails": len(hf), "n_adm": len(ha),
                    "recall_band": round(rec_b, 4), "precision_band": round(prec_b, 4)}

out = {
    "variant": "T219 = band-local upper-tail asymmetric upper-MAD EB-lite single-floor soft-score pooled-0 + 20% global anchor: med*=0.8*med_b+0.2*med_g; MADup*=w*MADup_b+(1-w)*MADup_g, w=n_b/(n_b+20); s=max(0,w191-med*)/(1.4826*max(MADup*,floor)); floor=max(5,0.5*MADup_g); p=1-exp(-s); per-band ROC TAU_p",
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [len(x) for x in cal_band],
              "cal_fails": [len(x) for x in cal_fails_band],
              "med_band": [round(v, 4) for v in med_band],
              "MED_global": round(MED_GLOB, 4),
              "anchor": ANCHOR,
              "center_star": [round(v, 4) for v in center_star],
              "MADup_band": [round(v, 4) for v in madup_band],
              "MADlow_band": [round(v, 4) for v in madlow_band],
              "MADsym_band": [round(v, 4) for v in madsym_band],
              "skew_up_over_low": [round(v, 4) if math.isfinite(v) else "inf" for v in skew_ratio],
              "MADUP_global": round(MADUP_GLOB, 4),
              "MADLOW_global": round(MADLOW_GLOB, 4),
              "MADsym_global": round(MAD_GLOB, 4),
              "n_band": n_band, "shrink_w": [round(v, 4) for v in wgt],
              "MADup_star": [round(v, 4) for v in madup_star],
              "floor_single": round(FLOOR1, 4),
              "scale_star": [round(v, 4) for v in scale_star],
              "floor_binds": floor_binds, "floor_frac": round(sum(floor_binds) / 3.0, 4),
              "collapse_raw_MADup": collapse, "collapse_frac": round(sum(collapse) / 3.0, 4)},
    "fit": {"tau_star_per_band": {str(b): tau_main[b] for b in range(3)},
            "TAU_STAR_band": ({str(b): round(TAUS[b], 4) for b in range(3)} if TAUS else None),
            "feasible": feas,
            "hard_fit_feasible": feas_hard,
            "hard_TAU_band": ({str(b): round(tau_hard[b]["TAU"], 4) for b in range(3)}
                              if feas_hard else None)},
    "finite_audit": {"nonfinite_or_neg_calib": nan_calib, "nonfinite_or_neg_held": nan_held,
                     "nonfinite_or_neg_arch": nan_arch,
                     "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0)},
    "mad0_slice": mad0_demo,
    "nsub_slice": sub_results,
    "lobo": {"per_leftout": lobo_pts, "var_pts": lobo_var, "pass_var_lt15": bool(lobo_var < 15)},
    "pooled1_ablation": p1,
    "pooled1_wins": bool(p1["held_recall_pts"] > pts),
    "ablations": {"anchor0_pooled0": abl_anchor0, "symmetricMAD_pooled0": abl_sym,
                  "anchor_delta_pts": round(pts - abl_anchor0["held_recall_pts"], 2),
                  "upper_vs_sym_delta_pts": round(pts - abl_sym["held_recall_pts"], 2)},
    "soft_vs_hard": {"rank_corr_spearman": rank_soft_hard,
                     "admit_agreement_held": round(agree, 4)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T219": e219, "cov_all": cov_all},
    "perband_held": band_held,
    "pooled_held": {"T219_pooled0": pe_main,
                    "recall_vs_T207_pts": round(pts - T207_REF, 2),
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe_main["p_fail_adm"]), 2)},
    "archive_adm_frac_T219": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e219["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"pts_ge_70": bool(pts >= 70), "heldB_keep_ge_70": bool(kp >= 70),
                      "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0),
                      "lobo_var_lt15": bool(lobo_var < 15),
                      "predicted_pts_75_85": pts, "predicted_keep": kp},
}
discard = (not feas or pts < 70 or kp < 70 or (pe_main["n_adm"] == 0)
           or not out["cov_ok"]
           or not (nan_calib == 0 and nan_held == 0 and nan_arch == 0))
out["degenerate_all_reject"] = bool(pe_main["n_adm"] == 0)
out["verdict"] = "DISCARD" if discard else "KEEP"
out["keep"] = bool(out["verdict"] == "KEEP" and out["runA_keep_cited"])
json.dump(out, open("results/aegis_v2/I2_r219_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
