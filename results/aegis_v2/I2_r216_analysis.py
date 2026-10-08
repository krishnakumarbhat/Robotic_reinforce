"""Run 223: T216 = hierarchical global-anchored signed EB-shrunk floored-denominator soft-score pooled-1 (director iter 41).

Director iter 41 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: zero-MAD blowup + over-localization; T213 signed+global-anchor (53.3) >> T214/T215 upper-tail local-only (100 discard).
Variation T216: hierarchical global-anchored signed EB-shrunk floored-denominator soft-score pooled-1.
Formula: med* = l*med_b+(1-l)*med_glob, l=n_b/(n_b+10)
Formula: MAD* = (n_b*MAD_b+10*MAD_glob)/(n_b+10), den=1.4826*max(MAD*, 0.2*MAD_glob, eps)
Formula: s = (w191-med*)/den signed, no max(0,.), pooled-1.
Validation: zero-MAD bands no longer infinite; sign retained; global floor stops collapse.
Validation: ablate pooled-1 vs pooled-0, check rank corr vs T213.
Predicted: 65-72pts, first keep >=70 candidate.
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
K_SHRINK = 10.0
FLOOR_FRAC = 0.2
PLATEAU = 46.67
T207_REF = 60.0
T213_REF = 53.33


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
cal_fails = [e for e in cal_eps if not e["success"]]
cal_fails_band = [[e for e in cal_band[b] if not e["success"]] for b in range(3)]

med_band, mad_band, n_band = [], [], []
for b in range(3):
    ws = [w191(e) for e in cal_band[b]]
    m = median(ws)
    med_band.append(m)
    mad_band.append(mad(ws, m))
    n_band.append(len(cal_band[b]))

w_all = [w191(e) for e in cal_eps]
MED_GLOB = median(w_all)
MAD_GLOB = mad(w_all, MED_GLOB)

lam_b = [n / (n + K_SHRINK) for n in n_band]
med_star = [lam_b[b] * med_band[b] + (1 - lam_b[b]) * MED_GLOB for b in range(3)]
mad_star_raw = [(n_band[b] * mad_band[b] + K_SHRINK * MAD_GLOB) / (n_band[b] + K_SHRINK) for b in range(3)]
floor_abs = FLOOR_FRAC * MAD_GLOB
mad_star = [max(v, floor_abs) for v in mad_star_raw]
floor_binds = [mad_star_raw[b] < floor_abs for b in range(3)]
scale_b = []
for b in range(3):
    sc = MAD_SCALE * max(mad_star[b], MAD_EPS)
    if not (sc > 1e-12):
        sc = 1.0
    scale_b.append(sc)

# T213 reference (same center, T213 denominator) for rank correlation
scale_t213 = []
for b in range(3):
    cand = [mad_band[b], MAD_GLOB / math.sqrt(n_band[b]), 0.5 * MAD_GLOB]
    sc = MAD_SCALE * max(cand) + MAD_EPS
    if not (sc > 1e-12):
        sc = 1.0
    scale_t213.append(sc)


def s_t216(e):
    b = band_of(e)
    return (w191(e) - med_star[b]) / scale_b[b]


def s_t213(e):
    b = band_of(e)
    return (w191(e) - med_star[b]) / scale_t213[b]


def pooled_stats(admit_fn, eps):
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if admit_fn(e)]
    caught = sum(1 for e in fails if not admit_fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


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


fit = fit_global(s_t216, cal_eps, cal_fails)
feas = fit["feasible"]
TAU = fit["TAU"] if feas else None
A = (lambda e: s_t216(e) >= TAU) if feas else (lambda e: False)


def fit_perband(score_fn):
    out = {}
    for b in range(3):
        fb = cal_fails_band[b]
        sc = sorted(score_fn(e) for e in fb)
        vals = []
        for q in Q_GRID:
            v = qceil(sc, q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        best = None
        for tv in vals:
            hb = cal_band[b]
            fails = [e for e in hb if not e["success"]]
            adm = [e for e in hb if score_fn(e) >= tv]
            rec = sum(1 for e in fails if not (score_fn(e) >= tv)) / len(fails)
            pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
            if rec >= 0.65 and (best is None or (pf, tv) < best[0]):
                best = ((pf, tv), tv, rec, pf)
        out[b] = {"TAU": best[1] if best else None, "feasible": best is not None,
                  "rec": round(best[2], 4) if best else None, "pf": round(best[3], 4) if best else None}
    return out


tau0 = fit_perband(s_t216)
feas0 = all(tau0[b]["feasible"] for b in range(3))
if feas0:
    _t0 = {b: tau0[b]["TAU"] for b in range(3)}
    A0 = lambda e, _t0=_t0: s_t216(e) >= _t0[band_of(e)]
else:
    A0 = lambda e: False

# zero-MAD synthetic check: MAD_b=0 -> MAD*=K*MAD_glob/(n+K), floored at 0.2*MAD_glob
eb0 = (K_SHRINK * MAD_GLOB) / (40 + K_SHRINK)
ms0 = max(eb0, floor_abs)
sc0 = MAD_SCALE * max(ms0, MAD_EPS)
s10_on = 10.0 / sc0
# floor-off comparison
sc0_off = MAD_SCALE * max(eb0, MAD_EPS)
s10_off = 10.0 / sc0_off
zero_finite = bool(math.isfinite(s10_on) and sc0 > 1e-12)
floor_caps = bool(s10_on < s10_off)

# sign retained: fraction of held episodes with s<0, and min score
held_s = [s_t216(e) for e in tst_eps]
sign_neg_frac = round(sum(1 for v in held_s if v < 0) / len(held_s), 4)
sign_min = round(min(held_s), 4)
sign_retained = bool(sign_min < 0)

rank_corr = round(spearman(held_s, [s_t213(e) for e in tst_eps]), 4)

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


e216 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe1 = pooled_eval(A)
pe0 = pooled_eval(A0)
pts = pe1["score_recall_pts"]
pts0 = pe0["score_recall_pts"]
kp = e216["keep_pct"]

band_held = {}
for b in range(3):
    hb = [e for e in tst_eps if band_of(e) == b]
    hf = [e for e in hb if not e["success"]]
    if not hf or not feas:
        band_held[b] = {"n": len(hb), "n_fails": len(hf)}
        continue
    caught = sum(1 for e in hf if not (s_t216(e) >= TAU))
    band_held[b] = {"n": len(hb), "n_fails": len(hf), "recall_band": round(caught / len(hf), 4)}

out = {
    "variant": "T216 = hierarchical global-anchored signed EB-shrunk floored-denominator soft-score pooled-1: s=(w191-[l*med_b+(1-l)*med_glob])/(1.4826*max((n_b*MAD_b+10*MAD_glob)/(n_b+10),0.2*MAD_glob)), l=n_b/(n_b+10); signed, no jerk term",
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [len(x) for x in cal_band],
              "cal_fails": [len(x) for x in cal_fails_band],
              "med_band": [round(v, 4) for v in med_band],
              "MAD_band": [round(v, 4) for v in mad_band],
              "MED_global": round(MED_GLOB, 4),
              "MAD_global": round(MAD_GLOB, 4),
              "n_band": n_band,
              "lambda_shrink": [round(v, 4) for v in lam_b],
              "med_star": [round(v, 4) for v in med_star],
              "MAD_star_raw": [round(v, 4) for v in mad_star_raw],
              "MAD_floor_abs": round(floor_abs, 4),
              "MAD_star": [round(v, 4) for v in mad_star],
              "scale_T216": [round(v, 4) for v in scale_b],
              "scale_T213ref": [round(v, 4) for v in scale_t213],
              "floor_binds": floor_binds, "floor_frac": round(sum(floor_binds) / 3.0, 4)},
    "fit": {"K_SHRINK": K_SHRINK, "FLOOR_FRAC": FLOOR_FRAC, "pooled": "pooled-1 global TAU",
            "TAU_STAR": round(TAU, 4) if TAU is not None else None,
            "calib_recall": fit.get("calib_recall"), "calib_pf": fit.get("calib_pf"),
            "feasible": feas,
            "abl_pooled0_perband": {str(b): tau0[b] for b in range(3)},
            "abl_pooled0_feasible": feas0},
    "perband_ROC_held_globalTAU": band_held,
    "zeroMAD_check": {"EB_at_MAD0": round(eb0, 4), "scale_floor_on": round(sc0, 4),
                      "scale_floor_off": round(sc0_off, 4),
                      "score_per_10units_on": round(s10_on, 4),
                      "score_per_10units_off": round(s10_off, 4),
                      "finite_no_blowup": zero_finite, "floor_caps": floor_caps,
                      "collapse_raw_MAD": [bool(v < 1e-9) for v in mad_band]},
    "sign_check": {"neg_frac_held": sign_neg_frac, "min_held_s": sign_min, "sign_retained": sign_retained},
    "rank_corr_vs_T213_held_spearman": rank_corr,
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T216": e216, "cov_all": cov_all, "T216_pooled0_abl": heldB_eval(A0)},
    "pooled_held": {"T216_pooled1": pe1, "T216_pooled0_abl": pe0,
                    "pooled1_vs_pooled0_pts": round(pts - pts0, 2),
                    "recall_vs_T207_pts": round(pts - T207_REF, 2),
                    "recall_vs_T213_pts": round(pts - T213_REF, 2),
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe1["p_fail_adm"]), 2)},
    "archive_adm_frac_T216": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e216["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"pts_ge_70": bool(feas and pts >= 70), "heldB_keep_ge_70": bool(kp >= 70),
                      "predicted_pts_65_72": pts, "predicted_keep": kp},
}
out["verdict"] = "KEEP" if (feas and pts >= 70 and kp >= 70 and out["cov_ok"]) else "DISCARD"
out["keep"] = bool(out["verdict"] == "KEEP" and out["runA_keep_cited"])
json.dump(out, open("results/aegis_v2/I2_r216_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
