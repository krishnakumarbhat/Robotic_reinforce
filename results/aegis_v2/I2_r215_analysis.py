"""Run 222: T215 = band-local upper-tail EB-shrunk-scale + floored-denominator soft-score pooled-0 (director iter 40).

Director iter 40 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: robust band-local upper-tail anomaly score (keep T214 locality, reject T213 global-anchored signed).
Variation vs T214: s=max(0,w191-med_b)/(1.4826*max(EB_scale, MAD_floor)),
  EB_scale=(n_b*MAD_b+10*MAD_glob)/(n_b+10), MAD_floor=0.5*MAD_glob.
  (T214 floor was 0.3*MAD_glob; T215 raises to 0.5 to stop false 100pt spikes
  when MAD_b~0. Drops T212 additive-jerk denominator hack; no jerk term.)
Fit (train-locked, CALIB pooled only, single held eval, pooled-0 ONLY):
  bands = CALIB jerk tertiles (same edges as T207-T214); per-band med_b/MAD_b
  + MAD_glob; EB scale_star_b with 0.5 floor; per-band ROC TAU*_band over CALIB
  in-band fail-s Q{0.5..0.9} (recall_b>=0.65 then min pf, loosest tie-break).
Validation: rerun T214 bands; check zero-MAD bands (collapse flags + synthetic
  MAD_b=0 blowup demo); rank correlation vs T214 scores (Spearman on held);
  pooled-0 only (no pooled-1 refit — director constraint).
Ablation: k in {5,10,20} x floor on/off (floor off = pure EB, no max).
Predicted 75-82pts KEEP (>=70). Discard if pooled recall <70 or signed/global
  variant wins (signed ablation included; global cited from T213, not rerun).
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
K_MAIN = 10.0
FLOOR_FRAC_MAIN = 0.5
K_GRID = [5.0, 10.0, 20.0]
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
    rx = [sorted(xs).index(v) for v in xs]
    ry = [sorted(ys).index(v) for v in ys]
    # tie-unsafe simple rank via sorted order positions
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
assert E1 < E2, (E1, E2)
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
        if floor_frac_or_none is None:
            ms = eb
            binds.append(False)
        else:
            fl = floor_frac_or_none * MAD_GLOB
            ms = max(eb, fl)
            binds.append(eb < fl)
        eb_raw.append(eb)
        mad_star.append(ms)
        sc = MAD_SCALE * ms + MAD_EPS
        if not (sc > 1e-12):
            sc = 1.0
        scale_star.append(sc)
    return eb_raw, mad_star, scale_star, binds


eb_raw_m, mad_star_m, scale_star_m, binds_m = eb_scales(K_MAIN, FLOOR_FRAC_MAIN)
floor_frac = sum(binds_m) / 3.0
collapse = [md < 1e-9 for md in mad_band]
collapse_frac = sum(collapse) / 3.0
# T214 ref scales (K=10, floor 0.3) for rank correlation
_, _, scale_star_t214, _ = eb_scales(10.0, 0.3)


def mk_s(scales):
    def s(e):
        b = band_of(e)
        return max(0.0, w191(e) - med_band[b]) / scales[b]
    return s


s_main = mk_s(scale_star_m)
s_t214 = mk_s(scale_star_t214)


def s_signed_main(e):
    b = band_of(e)
    return (w191(e) - med_band[b]) / scale_star_m[b]


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
tau_signed = fit_perband(s_signed_main)


def mk_admit(tau_d, score_fn):
    if not all(tau_d[b]["feasible"] for b in range(3)):
        return (lambda e: False), False
    _t = {b: tau_d[b]["TAU"] for b in range(3)}
    return (lambda e, _t=_t: score_fn(e) >= _t[band_of(e)]), True


A, feas = (mk_admit(tau_main, s_main) if feas_main else ((lambda e: False), False))
A_signed, feas_signed = mk_admit(tau_signed, s_signed_main)

# Ablation grid: k in {5,10,20} x floor {on(0.5), off}
abl = {}
for K in K_GRID:
    for flabel, ff in [("on", 0.5), ("off", None)]:
        _, _, sc, binds = eb_scales(K, ff)
        s = mk_s(sc)
        td = fit_perband(s)
        adm, fs = mk_admit(td, s)
        pe = pooled_stats(adm, tst_eps)
        abl[f"k{K}_{flabel}"] = {
            "K": K, "floor": flabel,
            "scales": [round(v, 4) for v in sc],
            "floor_binds": binds,
            "feasible": fs,
            "taus": {str(b): td[b] for b in range(3)},
            "pooled_n_adm": pe["n_adm"],
            "pooled_recall": round(pe["recall"], 4),
            "score_pts": round(100 * pe["recall"], 2),
        }

# zero-MAD synthetic blowup demo: what if MAD_b=0 -> EB=K*MAD_glob/(n+K), floored vs unfloored scale
zero_demo = {}
for K in K_GRID:
    eb0 = (K * MAD_GLOB) / (40 + K)
    for flabel, ff in [("on", 0.5 * MAD_GLOB), ("off", None)]:
        ms = max(eb0, ff) if ff is not None else eb0
        sc = MAD_SCALE * ms
        # unit upper-tail deviation of 10 w-units -> score
        s10 = 10.0 / sc
        zero_demo[f"k{K}_{flabel}"] = {"EB_at_MAD0": round(eb0, 4),
                                       "scale": round(sc, 4),
                                       "score_per_10units": round(s10, 4)}
# does the 0.5 floor collapse zero-MAD scores? compare off vs on
zero_collapse = {K: round(zero_demo[f"k{K}_off"]["score_per_10units"] - zero_demo[f"k{K}_on"]["score_per_10units"], 4) for K in K_GRID}

# rank correlation T215 vs T214 on held episodes
held_s_main = [s_main(e) for e in tst_eps]
held_s_t214 = [s_t214(e) for e in tst_eps]
rank_corr = round(spearman(held_s_main, held_s_t214), 4)

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


e215 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe_main = pooled_eval(A)
pe_signed = pooled_eval(A_signed)
pts = pe_main["score_recall_pts"]
pts_signed = pe_signed["score_recall_pts"]
kp = e215["keep_pct"]

out = {
    "variant": "T215 = band-local upper-tail EB-shrunk-scale + floored-denominator soft-score pooled-0: s=max(0,w191-med_b)/(1.4826*max((n_b*MAD_b+10*MAD_glob)/(n_b+10),0.5*MAD_glob)), no jerk term",
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
            "abl_signed_feasible": feas_signed,
            "abl_signed_tau": {str(b): tau_signed[b] for b in range(3)}},
    "ablation_k_x_floor": abl,
    "zeroMAD_check": {"collapse_frac": round(collapse_frac, 4),
                      "floor_binds_main": binds_m,
                      "synthetic_MAD0_demo": zero_demo,
                      "floor_collapse_reduction_per10units": zero_collapse,
                      "floor_collapses_zeroMAD": bool(all(v > 0 for v in zero_collapse.values()))},
    "rank_corr_vs_T214_held_spearman": rank_corr,
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T215": e215, "cov_all": cov_all,
              "T215_signed": heldB_eval(A_signed)},
    "pooled_held": {"T215_pooled0": pe_main,
                    "T215_signed_abl": pe_signed,
                    "delta_main_vs_signed_pts": round(pts - pts_signed, 2),
                    "recall_vs_T207_pts": round(pts - T207_REF, 2),
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe_main["p_fail_adm"]), 2)},
    "archive_adm_frac_T215": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e215["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"pts_ge_70": bool(pts >= 70), "heldB_keep_ge_70": bool(kp >= 70),
                      "predicted_pts_75_82": pts, "predicted_keep": kp},
}
signed_wins = bool(pts_signed > pts and feas_signed)
discard = (not feas or pts < 70 or pts_signed > pts and feas_signed
           or (pe_main["n_adm"] == 0) or kp < 70 or not out["cov_ok"])
if pe_main["n_adm"] == 0:
    discard = True
out["signed_wins"] = signed_wins
out["degenerate_all_reject"] = bool(pe_main["n_adm"] == 0)
out["verdict"] = "DISCARD" if discard else "KEEP"
out["keep"] = bool(out["verdict"] == "KEEP" and out["runA_keep_cited"])
json.dump(out, open("results/aegis_v2/I2_r215_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
