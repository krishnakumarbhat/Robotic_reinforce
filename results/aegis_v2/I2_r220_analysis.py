"""Run 227: T220 = cross-band pooled two-tail IQR soft-score, pooled-8, 50% global
anchor (director iter 45, single-brain fallback
opencode-responses/muse-spark-1.3-contributor-free).

Frontier: leave band-local upper-tail MAD (0/3 keeps, iters 42-44 saturated:
T217/T218/T219 all bands-0+1-infeasible -> degenerate all-reject 100.0).
Variation vs T207-T219: cross-band pooled scale + two-tail absolute + IQR sigma.
Formula:
  center_b = 0.5*med_b + 0.5*med_g          (50% global anchor, per jerk band)
  IQR_pool = Q75 - Q25 over ALL CALIB w191  (cross-band pooled, qceil convention)
  floor_g  = 5.0                             (absolute w191-unit guard)
  s_raw    = |w191 - center_b| / (0.7413*max(IQR_pool, floor_g))
  s        = min(s_raw, 6.0)                 (soft-clip at 6)
 Admit iff s <= TAU* (two-tail: central admits, tails reject). Single global TAU*
fit on CALIB pooled over 8-point success-quantile grid Q{0.5..0.9} ("pooled-8":
one threshold, 8 candidates; recall>=0.65 then min P(fail|admit), loosest tie).
Bands = CALIB jerk tertiles (centers only; scale is pooled). Frozen I7+I10.
Zero rig edits, frozen R173 files. G7-clean, seg 15.
Validation: rescore same HELD set as T219's 100.0 degenerate + leave-one-band-out
(1 held-out band x3). Line-KEEP iff pts<70 AND non-degenerate AND finite AND
cov_ok (director rule). Predicted 55+/-12pts. ABORT-IF pts>=70 -> kill robust-z
family, next rank/quantile.
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
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]  # 8 candidates = pooled-8
IQR_SIGMA = 0.7413  # sigma ~= IQR/1.349; 1/1.349 = 0.7413
FLOOR_G = 5.0
CLIP = 6.0
ANCHOR = 0.5
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


def iqr_linear(xs):
    s = sorted(xs)
    n = len(s)
    def qp(q):
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)
    return qp(0.75) - qp(0.25)


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
assert len(cal_fails) == 17, len(cal_fails)

w_all = [w191(e) for e in cal_eps]
MED_GLOB = median(w_all)
med_band = [median([w191(e) for e in cal_band[b]]) for b in range(3)]
IQR_POOL = qceil(w_all, 0.75) - qceil(w_all, 0.25)
IQR_LIN = iqr_linear(w_all)
SCALE = IQR_SIGMA * max(IQR_POOL, FLOOR_G)
floor_binds = IQR_POOL < FLOOR_G
center = [(1 - ANCHOR) * med_band[b] + ANCHOR * MED_GLOB for b in range(3)]


def s_raw(e):
    return abs(w191(e) - center[band_of(e)]) / SCALE


def s_cli(e):
    return min(s_raw(e), CLIP)


def pooled_stats(admit_fn, eps):
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if admit_fn(e)]
    caught = sum(1 for e in fails if not admit_fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


# --- pooled-8 fit: single global TAU* over 8 success-quantile candidates ---
# (conformal direction for a reject-score admitted class-by-construction:
#  TAU = Qq over CALIB SUCCESS s; admit iff s<=TAU. Fail-quantile grid would
#  invert the operating point for admit-<=; successes are the reference class.)
cal_succ = [e for e in cal_eps if e["success"]]
cand = sorted(s_cli(e) for e in cal_succ)
vals = []
for q in Q_GRID:
    v = qceil(cand, q)
    if not vals or abs(v - vals[-1]) > 1e-12:
        vals.append(v)
best = None
for tv in vals:
    stt = pooled_stats(lambda e, tv=tv: s_cli(e) <= tv, cal_eps)
    if stt["recall"] >= 0.65:
        key = (stt["pf"], -tv)  # min pf, then loosest (largest TAU)
        if best is None or key < best[0]:
            best = (key, tv, stt)
feasible = best is not None
TAU_STAR = best[1] if feasible else None
A = (lambda e, tv=TAU_STAR: s_cli(e) <= tv) if feasible else (lambda e: False)


def audit_finite(fn, eps):
    bad = 0
    for e in eps:
        v = fn(e)
        if not (math.isfinite(v) and v >= 0):
            bad += 1
    return bad


nan_calib = audit_finite(s_cli, cal_eps)
nan_held = audit_finite(s_cli, tst_eps)
nan_arch = audit_finite(s_cli, arc_eps)

# --- Stress 1: zero-IQR synthetic (floor must cap blowup) ---
zero_scale = IQR_SIGMA * max(0.0, FLOOR_G)
mad0_demo = {"IQR_set0": True, "scale": round(zero_scale, 4),
             "s_raw_per_10units": round(10.0 / zero_scale, 4),
             "s_clip_per_10units": round(min(10.0 / zero_scale, CLIP), 4),
             "finite": bool(math.isfinite(10.0 / zero_scale))}

# --- Stress 2: LOBO (1 held-out band x3, global refit on 2 bands) ---
lobo_pts = {}
for left in range(3):
    pool = [e for b in range(3) for e in cal_band[b] if b != left]
    fails = sorted(s_cli(e) for e in pool if e["success"])
    vs = []
    for q in Q_GRID:
        v = qceil(fails, q)
        if not vs or abs(v - vs[-1]) > 1e-12:
            vs.append(v)
    b2 = None
    for tv in vs:
        s2 = pooled_stats(lambda e, tv=tv: s_cli(e) <= tv, pool)
        if s2["recall"] >= 0.65:
            key = (s2["pf"], -tv)
            if b2 is None or key < b2[0]:
                b2 = (key, tv, s2)
    if b2 is None:
        lobo_pts[str(left)] = {"feasible": False, "held_recall_pts": 0.0}
    else:
        tv = b2[1]
        ps = pooled_stats(lambda e, tv=tv: s_cli(e) <= tv, tst_eps)
        lobo_pts[str(left)] = {"feasible": True,
                               "held_recall_pts": round(100 * ps["recall"], 2),
                               "TAU": round(tv, 4)}
lobo_vals = [lobo_pts[str(b)]["held_recall_pts"] for b in range(3)]
lobo_var = round(max(lobo_vals) - min(lobo_vals), 2)

# --- Ablations (same pooled-8 protocol) ---
def abl_eval(score_fn):
    cand2 = sorted(score_fn(e) for e in cal_succ)
    vs = []
    for q in Q_GRID:
        v = qceil(cand2, q)
        if not vs or abs(v - vs[-1]) > 1e-12:
            vs.append(v)
    b2 = None
    for tv in vs:
        s2 = pooled_stats(lambda e, tv=tv: score_fn(e) <= tv, cal_eps)
        if s2["recall"] >= 0.65:
            key = (s2["pf"], -tv)
            if b2 is None or key < b2[0]:
                b2 = (key, tv, s2)
    if b2 is None:
        fn = lambda e: False
        ok, tau = False, None
    else:
        tau = b2[1]
        fn = lambda e, tau=tau: score_fn(e) <= tau
        ok = True
    ps = pooled_stats(fn, tst_eps)
    return {"feasible": ok, "TAU": round(tau, 4) if ok else None,
            "held_recall_pts": round(100 * ps["recall"], 2), "n_adm": ps["n_adm"]}


def mk_two_tail(anchor, scale, clip):
    c = [(1 - anchor) * med_band[b] + anchor * MED_GLOB for b in range(3)]
    def fn(e):
        return min(abs(w191(e) - c[band_of(e)]) / scale, clip)
    return fn


MAD_GLOB = median([abs(x - MED_GLOB) for x in w_all])
mad_scale = 1.4826 * max(MAD_GLOB, FLOOR_G)
abl_anchor0 = abl_eval(mk_two_tail(0.0, SCALE, CLIP))
abl_anchor1 = abl_eval(mk_two_tail(1.0, SCALE, CLIP))
abl_mad = abl_eval(mk_two_tail(ANCHOR, mad_scale, CLIP))
abl_noclip = abl_eval(mk_two_tail(ANCHOR, SCALE, float("inf")))
abl_upper = abl_eval(lambda e: min(max(0.0, w191(e) - center[band_of(e)]) / SCALE, CLIP))

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


e220 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = pooled_eval(A)
pts = pe["score_recall_pts"]
kp = e220["keep_pct"]
smax_held = max(s_raw(e) for e in tst_eps)
clip_uses = sum(1 for e in tst_eps if s_raw(e) > CLIP)

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
    "variant": "T220 = cross-band pooled two-tail IQR soft-score pooled-8 + 50% global anchor: s=min(|w191-[0.5*med_b+0.5*med_g]|/(0.7413*max(IQR_pool,floor_g)),6); admit iff s<=TAU* (single global TAU, 8-pt success-quantile grid)",
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [40, 40, 40],
              "cal_fails": [6, 4, 7],
              "med_band": [round(v, 4) for v in med_band],
              "MED_global": round(MED_GLOB, 4),
              "anchor": ANCHOR,
              "center_star": [round(v, 4) for v in center],
              "IQR_pool_qceil": round(IQR_POOL, 4),
              "IQR_pool_linear": round(IQR_LIN, 4),
              "floor_g": FLOOR_G,
              "scale": round(SCALE, 4),
              "floor_binds": floor_binds,
              "MAD_global": round(MAD_GLOB, 4),
              "mad_scale": round(mad_scale, 4),
              "clip": CLIP,
              "smax_raw_held": round(smax_held, 4),
              "clip_uses_held": clip_uses},
    "fit": {"TAU_STAR": round(TAU_STAR, 4) if feasible else None,
            "feasible": feasible,
            "calib_recall": round(best[2]["recall"], 4) if feasible else None,
            "calib_pf": round(best[2]["pf"], 4) if feasible else None,
            "grid_candidates": [round(v, 4) for v in vals]},
    "finite_audit": {"nonfinite_or_neg_calib": nan_calib,
                     "nonfinite_or_neg_held": nan_held,
                     "nonfinite_or_neg_arch": nan_arch,
                     "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0)},
    "iqr0_slice": mad0_demo,
    "lobo": {"per_leftout": lobo_pts, "var_pts": lobo_var,
             "pass_var_lt15": bool(lobo_var < 15)},
    "ablations": {"anchor0_local": abl_anchor0, "anchor1_global": abl_anchor1,
                  "mad_scale": abl_mad, "no_clip": abl_noclip,
                  "upper_only": abl_upper,
                  "anchor_delta_vs_main_pts": {
                      "a0": round(pts - abl_anchor0["held_recall_pts"], 2),
                      "a1": round(pts - abl_anchor1["held_recall_pts"], 2)},
                  "iqr_vs_mad_pts": round(pts - abl_mad["held_recall_pts"], 2),
                  "clip_vs_noclip_pts": round(pts - abl_noclip["held_recall_pts"], 2),
                  "twotail_vs_upper_pts": round(pts - abl_upper["held_recall_pts"], 2)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T220": e220, "cov_all": cov_all},
    "perband_held": band_held,
    "pooled_held": {"T220_pooled8": pe,
                    "recall_vs_T207_pts": round(pts - T207_REF, 2),
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["p_fail_adm"]), 2)},
    "archive_adm_frac_T220": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e220["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"pts_lt_70": bool(pts < 70),
                      "non_degenerate": bool(pe["n_adm"] > 0),
                      "zero_inf_nan": bool(nan_calib == 0 and nan_held == 0 and nan_arch == 0),
                      "lobo_var_lt15": bool(lobo_var < 15),
                      "predicted_pts_55pm12": pts, "predicted_keep": kp},
}
line_keep = (feasible and pts < 70 and pe["n_adm"] > 0
             and out["cov_ok"]
             and (nan_calib == 0 and nan_held == 0 and nan_arch == 0))
out["degenerate_all_reject"] = bool(pe["n_adm"] == 0)
out["line_keep"] = bool(line_keep)
out["abort_if_fired"] = bool(not line_keep)
out["verdict"] = "KEEP-LINE" if line_keep else "DISCARD"
out["keep"] = False  # post-hoc rescore can never set physical keep (G2/G4)
json.dump(out, open("results/aegis_v2/I2_r220_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
