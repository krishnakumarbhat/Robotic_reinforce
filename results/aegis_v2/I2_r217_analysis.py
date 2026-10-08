"""Run 224: T217 = band-local upper-tail EB-shrunk floored-denominator pooled-0 (director iter 42).

Director iter 42 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: T215 band-local upper-tail EB-shrunk floored-denominator pooled-0 >> T216 hierarchical-signed pooled-1.
Variation T217 = T215 + keep local med_b, keep upper-tail, fix scale floor only.
Formula: s=max(0,w191-med_b)/(1.4826*max(MAD_EB_b, 0.15*MAD_global, eps)),
  MAD_EB_b=(n_b*MAD_b+10*M_global)/(n_b+10).
Change vs T214/T215: floor tied to global scale (0.15*M_global), not 10*M count term alone; no median shrinkage.
Why: T216 proves global-anchored median + signed + pooled-1 destroys signal; discards on T214/T215 are
  denominator-collapse on MAD_b=0 bands (upper-tail max(0,.) + tiny scale -> infeasible bands).
Validation: rerun keep>=70 suite + MAD_b=0 slice + upper-tail precision; must have zero inf/NaN s.
Predicted: 78-88pts, keep (vs 100pts-discard unstable, 53pts-stable-bad).
Next if fail: T218 = T217 + winsorize w191 at p99.5 before scoring.
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
EXP_THETA_MARG = 0.014133
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
MAD_SCALE = 1.4826
MAD_EPS = 1e-9
K_MAIN = 10.0
FLOOR_FRAC_MAIN = 0.15
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


def eb_scales(K, floor_frac_or_none):
    eb_raw, mad_star, scale_star, binds = [], [], [], []
    for b in range(3):
        eb = (n_band[b] * mad_band[b] + K * MAD_GLOB) / (n_band[b] + K)
        cands = [eb, MAD_EPS]
        if floor_frac_or_none is not None:
            fl = floor_frac_or_none * MAD_GLOB
            cands.append(fl)
            binds.append(eb < fl)
        else:
            binds.append(False)
        ms = max(cands)
        eb_raw.append(eb)
        mad_star.append(ms)
        sc = MAD_SCALE * ms
        if not (sc > 1e-12):
            sc = 1.0
        scale_star.append(sc)
    return eb_raw, mad_star, scale_star, binds


eb_raw_m, mad_star_m, scale_star_m, binds_m = eb_scales(K_MAIN, FLOOR_FRAC_MAIN)
floor_frac = sum(binds_m) / 3.0
collapse = [md < 1e-9 for md in mad_band]
collapse_frac = sum(collapse) / 3.0
# T215 ref scales (K=10, floor 0.5) for rank correlation
_, _, scale_star_t215, _ = eb_scales(10.0, 0.5)
# raw-MAD ref (no EB, no floor) to quantify EB+floor delta
scale_raw = []
for b in range(3):
    sc = MAD_SCALE * max(mad_band[b], MAD_EPS)
    if not (sc > 1e-12):
        sc = 1.0
    scale_raw.append(sc)


def mk_s(scales):
    def s(e):
        b = band_of(e)
        return max(0.0, w191(e) - med_band[b]) / scales[b]
    return s


s_main = mk_s(scale_star_m)
s_t215 = mk_s(scale_star_t215)
s_raw = mk_s(scale_raw)


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
tau_raw = fit_perband(s_raw)


def mk_admit(tau_d, score_fn):
    if not all(tau_d[b]["feasible"] for b in range(3)):
        return (lambda e: False), False
    _t = {b: tau_d[b]["TAU"] for b in range(3)}
    return (lambda e, _t=_t: score_fn(e) >= _t[band_of(e)]), True


A, feas = (mk_admit(tau_main, s_main) if feas_main else ((lambda e: False), False))
A_raw, feas_raw = mk_admit(tau_raw, s_raw)

# zero inf/NaN audit on CALIB+HELD+ARCH scores
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

# MAD_b=0 slice: synthetic — what if a band MAD collapses to 0
zero_demo = {}
eb0 = (K_MAIN * MAD_GLOB) / (40 + K_MAIN)
for flabel, ff in [("on", 0.15 * MAD_GLOB), ("off", None)]:
    cands = [eb0, MAD_EPS] + ([ff] if ff is not None else [])
    ms = max(cands)
    sc = MAD_SCALE * ms
    zero_demo[flabel] = {"EB_at_MAD0": round(eb0, 4), "scale": round(sc, 4),
                         "score_per_10units": round(10.0 / sc, 4),
                         "finite": bool(math.isfinite(10.0 / sc))}
zero_collapse_reduction = round(zero_demo["off"]["score_per_10units"] - zero_demo["on"]["score_per_10units"], 4)

# upper-tail precision: precision among admitted (pooled held) + per-band in-band fail precision
held_s_main = [s_main(e) for e in tst_eps]
held_s_t215 = [s_t215(e) for e in tst_eps]
rank_corr = round(spearman(held_s_main, held_s_t215), 4)

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


e217 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe_main = pooled_eval(A)
pe_raw = pooled_eval(A_raw)
pts = pe_main["score_recall_pts"]
pts_raw = pe_raw["score_recall_pts"]
kp = e217["keep_pct"]

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
    "variant": "T217 = band-local upper-tail EB-shrunk floored-denominator pooled-0: s=max(0,w191-med_b)/(1.4826*max((n_b*MAD_b+10*MAD_glob)/(n_b+10),0.15*MAD_glob,eps)), local med, upper-tail, no median shrinkage",
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [len(x) for x in cal_band],
              "cal_fails": [len(x) for x in cal_fails_band],
              "med_band": [round(v, 4) for v in med_band],
              "MAD_band": [round(v, 4) for v in mad_band],
              "MAD_global": round(MAD_GLOB, 4),
              "MED_global": round(MED_GLOB, 4),
              "n_band": n_band,
              "K_main": K_MAIN, "FLOOR_FRAC": FLOOR_FRAC_MAIN,
              "MAD_floor": round(FLOOR_FRAC_MAIN * MAD_GLOB, 4),
              "EB_raw": [round(v, 4) for v in eb_raw_m],
              "MAD_star": [round(v, 4) for v in mad_star_m],
              "scale_star": [round(v, 4) for v in scale_star_m],
              "floor_binds": binds_m, "floor_frac": round(floor_frac, 4),
              "collapse_raw_MAD": collapse, "collapse_frac": round(collapse_frac, 4)},
    "fit": {"tau_star_per_band": {str(b): tau_main[b] for b in range(3)},
            "TAU_STAR_band": ({str(b): round(TAUS[b], 4) for b in range(3)} if TAUS else None),
            "feasible": feas,
            "abl_rawMAD_feasible": feas_raw,
            "abl_rawMAD_tau": {str(b): tau_raw[b] for b in range(3)}},
    "finite_audit": {"nonfinite_or_neg_calib": nan_calib, "nonfinite_or_neg_held": nan_held,
                     "nonfinite_or_neg_arch": nan_arch,
                     "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0)},
    "zeroMAD_slice": {"collapse_frac": round(collapse_frac, 4),
                      "floor_binds_main": binds_m,
                      "synthetic_MAD0": zero_demo,
                      "floor_reduction_per10units": zero_collapse_reduction},
    "rank_corr_vs_T215_held_spearman": rank_corr,
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T217": e217, "cov_all": cov_all, "T217_rawMAD": heldB_eval(A_raw)},
    "perband_held": band_held,
    "pooled_held": {"T217_pooled0": pe_main,
                    "T217_rawMAD_abl": pe_raw,
                    "delta_main_vs_raw_pts": round(pts - pts_raw, 2),
                    "recall_vs_T207_pts": round(pts - T207_REF, 2),
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe_main["p_fail_adm"]), 2)},
    "archive_adm_frac_T217": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e217["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"pts_ge_70": bool(pts >= 70), "heldB_keep_ge_70": bool(kp >= 70),
                      "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0),
                      "predicted_pts_78_88": pts, "predicted_keep": kp},
}
discard = (not feas or pts < 70 or kp < 70 or (pe_main["n_adm"] == 0)
           or not out["cov_ok"]
           or not (nan_calib == 0 and nan_held == 0 and nan_arch == 0))
out["degenerate_all_reject"] = bool(pe_main["n_adm"] == 0)
out["verdict"] = "DISCARD" if discard else "KEEP"
out["keep"] = bool(out["verdict"] == "KEEP" and out["runA_keep_cited"])
json.dump(out, open("results/aegis_v2/I2_r217_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
