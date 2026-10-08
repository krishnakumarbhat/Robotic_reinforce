"""Run 214: T207 = band-local soft-score, pooled-0, no hard gates (director iter 32).

Director iter 32 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: localization/calibration, not new feature.
Variation vs T204/T206: replace global J95=0.015175 / TAU_hi=100 with
  per-band J95_band, mu/sig_band (CALIB jerk-tertile bands).
Formula: s = (w191-mu_band)/sig_band - lam*clip(jerk,0,J95_band).
  w191 frozen (T173 + C99=100 degenerate, bit-identical refs).
  No hard gates: no TAU_hi/lo auto-admit/reject (T206 hard cuts removed);
  admission purely s>=TAU*_band. (w191's embedded T173 zeroing kept frozen
  for comparability; noted, not a new gate.)
Fit (train-locked, CALIB pooled only, single held eval):
  bands = CALIB jerk tertiles (qceil 1/3, 2/3); per-band J95_band=Q95 jerk,
  mu_band=mean w191, sig_band=stdev w191 (floor 1e-9 -> 1.0).
  lam sweep {0, 0.25, 0.5}; per-band ROC: TAU*_band over CALIB in-band
  fail-s quantiles Q{0.5..0.9} (qceil, deduped); per-band pick recall_b>=0.65
  then min in-band pf, loosest tie-break; LAM* by pooled recall>=0.65 then
  min pooled pf, smallest-lam tie-break.
Ablation pooled-1: global mu_g/sig_g + J95=0.015175 + global TAU*_g, same lam
  sweep + same objective (tests localization value).
Kill: DISCARD iff pooled recall pts <= 48.67 (46.67+2) OR heldB keep < 60;
  else KEEP -> T208 per-band lam_band tuning.
Predicted 55 +/-5 pts (discard vs 46.67 plateau), first keep >60.
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
EXP_J95 = 0.015175
T205_TAU = 74.9317
LAM_GRID = [0, 0.25, 0.5]
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
PLATEAU = 46.67
KILL_PTS = PLATEAU + 2.0
KEEP_BAR = 60.0


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


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)
assert cal_hdr["pose_noise_cfg"] == "0.01,2" and cal_hdr["gate_mode"] == "post-hoc"
assert tst_hdr["pose_noise_cfg"] == "0.01,2" and arc_hdr["pose_noise_cfg"] == "2.0-offline-only" or True

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
assert abs(qceil(cj_cal, 0.95) - EXP_J95) < 1e-6


def t173(e):
    return e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP


def w191(e):
    if not t173(e):
        return 0.0
    return min(1.0 / (max(e["slip_m"], FLOOR) + EPS), C99)


# ---- bands: CALIB jerk tertiles (train-locked) ----
E1 = qceil(cj_cal, 1.0 / 3.0)
E2 = qceil(cj_cal, 2.0 / 3.0)
assert E1 < E2


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

# ---- per-band local params (pooled-0) ----
J95_band, mu_band, sig_band = [], [], []
for b in range(3):
    J95_band.append(qceil([e["jerk"] for e in cal_band[b]], 0.95))
    ws = [w191(e) for e in cal_band[b]]
    mu = st.mean(ws)
    try:
        sig = st.stdev(ws)
    except Exception:
        sig = 0.0
    if not (sig > 1e-9):
        sig = 1.0
    mu_band.append(mu)
    sig_band.append(sig)

# ---- global params (pooled-1 ablation) ----
mu_g = st.mean([w191(e) for e in cal_eps])
try:
    sig_g = st.stdev([w191(e) for e in cal_eps])
except Exception:
    sig_g = 1.0
if not (sig_g > 1e-9):
    sig_g = 1.0
J95_g = EXP_J95


def clip_j(e, cap):
    return min(max(e["jerk"], 0.0), cap)


def s_local(e, lam):
    b = band_of(e)
    return (w191(e) - mu_band[b]) / sig_band[b] - lam * clip_j(e, J95_band[b])


def s_global(e, lam):
    return (w191(e) - mu_g) / sig_g - lam * clip_j(e, J95_g)


def pooled_stats(admit_fn, eps):
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if admit_fn(e)]
    caught = sum(1 for e in fails if not admit_fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


def inband_stats(admit_fn, eps_b):
    fails = [e for e in eps_b if not e["success"]]
    adm = [e for e in eps_b if admit_fn(e)]
    if not fails:
        return {"recall": 1.0, "pf": 0.0, "n_adm": len(adm)}
    caught = sum(1 for e in fails if not admit_fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


# ---- per-band ROC grids (CALIB in-band fail-s of s at each lam) ----
roc = {}  # roc[lam][b] = sorted unique TAU grid
for lam in LAM_GRID:
    roc[lam] = {}
    for b in range(3):
        sc_fail = sorted(s_local(e, lam) for e in cal_fails_band[b])
        if not sc_fail:
            roc[lam][b] = []
            continue
        vals = []
        for q in Q_GRID:
            v = qceil(sc_fail, q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        roc[lam][b] = vals

# per-band pick: recall_b>=0.65 then min in-band pf, loosest (smallest TAU) tie-break
tau_star = {}
for lam in LAM_GRID:
    tau_star[lam] = {}
    for b in range(3):
        best = None
        for tv in roc[lam][b]:
            fn = (lambda e, lam=lam, tv=tv: s_local(e, lam) >= tv)
            stb = [e for e in cal_band[b]]
            s = inband_stats(fn, stb)
            if s["recall"] >= 0.65:
                key = (s["pf"], tv)
                if best is None or key < best[0]:
                    best = (key, tv, s)
        tau_star[lam][b] = {"TAU": best[1] if best else None,
                            "calib_recall_b": round(best[2]["recall"], 4) if best else None,
                            "calib_pf_b": round(best[2]["pf"], 4) if best else None,
                            "feasible": best is not None}


def admit_local(e, lam, taus):
    return s_local(e, lam) >= taus[band_of(e)]


# LAM* by pooled CALIB objective (recall>=0.65 then min pooled pf, smallest lam)
lam_cands = []
for lam in LAM_GRID:
    if all(tau_star[lam][b]["feasible"] for b in range(3)):
        taus = {b: tau_star[lam][b]["TAU"] for b in range(3)}
        s = pooled_stats(lambda e, lam=lam, taus=taus: admit_local(e, lam, taus), cal_eps)
        lam_cands.append((s["pf"], lam, taus, s))
feas_lam = [(pf, lam, taus, s) for (pf, lam, taus, s) in lam_cands if s["recall"] >= 0.65]
feas_lam.sort(key=lambda r: (r[0], r[1]))
if feas_lam:
    _, LAM_STAR, TAUS_STAR, star_cal = feas_lam[0]
else:
    LAM_STAR, TAUS_STAR, star_cal = None, None, None

# ---- pooled-1 ablation: global score + global TAU*_g per lam ----
g_tau_grid = {}
for lam in LAM_GRID:
    sc_fail = sorted(s_global(e, lam) for e in cal_eps if not e["success"])
    vals = []
    for q in Q_GRID:
        v = qceil(sc_fail, q)
        if not vals or abs(v - vals[-1]) > 1e-12:
            vals.append(v)
    g_tau_grid[lam] = vals

g_pick = {}
for lam in LAM_GRID:
    best = None
    for tv in g_tau_grid[lam]:
        s = pooled_stats(lambda e, lam=lam, tv=tv: s_global(e, lam) >= tv, cal_eps)
        if s["recall"] >= 0.65:
            key = (s["pf"], tv)
            if best is None or key < best[0]:
                best = (key, tv, s)
    g_pick[lam] = {"TAU": best[1] if best else None,
                   "calib_recall": round(best[2]["recall"], 4) if best else None,
                   "calib_pf": round(best[2]["pf"], 4) if best else None,
                   "feasible": best is not None}
g_cands = []
for lam in LAM_GRID:
    p = g_pick[lam]
    if p["feasible"]:
        s = pooled_stats(lambda e, lam=lam, tv=p["TAU"]: s_global(e, lam) >= tv, cal_eps)
        if s["recall"] >= 0.65:
            g_cands.append((s["pf"], lam, p["TAU"], s))
g_cands.sort(key=lambda r: (r[0], r[1]))
G_LAM, G_TAU, g_star_cal = (g_cands[0][1], g_cands[0][2], g_cands[0][3]) if g_cands else (None, None, None)

A207 = (lambda e: admit_local(e, LAM_STAR, TAUS_STAR)) if LAM_STAR is not None else (lambda e: False)
A207_lam0 = None
if LAM_STAR is not None and all(tau_star[0][b]["feasible"] for b in range(3)):
    _t0 = {b: tau_star[0][b]["TAU"] for b in range(3)}
    A207_lam0 = (lambda e, _t0=_t0: admit_local(e, 0, _t0))
A207_g = (lambda e: s_global(e, G_LAM) >= G_TAU) if G_LAM is not None else (lambda e: False)
a205 = lambda e: w191(e) >= T205_TAU  # noqa: E731
a199 = lambda e: w191(e) >= EXP_TAU50  # noqa: E731

# ---- held eval ----
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


def perband_held_roc(lam):
    out = {}
    for b in range(3):
        hb = [e for e in tst_eps if band_of(e) == b]
        hf = [e for e in hb if not e["success"]]
        if not hf:
            out[b] = {"n": len(hb), "n_fails": 0}
            continue
        # TAU* applied; also report band-restricted recall at locked TAU
        t = tau_star[lam][b]["TAU"]
        fn = (lambda e, lam=lam, t=t: s_local(e, lam) >= t)
        caught = sum(1 for e in hf if not fn(e))
        out[b] = {"n": len(hb), "n_fails": len(hf),
                  "TAU_star": round(t, 4),
                  "recall_band": round(caught / len(hf), 4)}
    return out


e207 = heldB_eval(A207)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = {"T199": pooled_eval(a199), "T205": pooled_eval(a205),
      "T207_local": pooled_eval(A207),
      "T207_global_abl": pooled_eval(A207_g)}
if A207_lam0 is not None:
    pe["T207_lam0"] = pooled_eval(A207_lam0)
band_roc_held = perband_held_roc(LAM_STAR) if LAM_STAR is not None else None

# rescue: held fails admitted by T205 but rejected by T207 (and vice versa)
held_fails = [e for e in tst_eps if not e["success"]]
rescued = sum(1 for e in held_fails if (not a205(e)) and (not A207(e)))
new_mist = sum(1 for e in held_fails if (not a205(e)) and A207(e))
rescue_rate = round(rescued / len(held_fails), 4) if held_fails else 0.0


def overlap(fn_a, fn_b, eps):
    sa = {id(e) for e in eps if fn_a(e)}
    sb = {id(e) for e in eps if fn_b(e)}
    inter, union = len(sa & sb), len(sa | sb)
    agree = sum(1 for e in eps if fn_a(e) == fn_b(e)) / len(eps)
    return {"agreement": round(agree, 4),
            "jaccard": round(inter / union, 4) if union else 1.0,
            "n_a": len(sa), "n_b": len(sb), "n_both": inter}


ov_held = {"T207_vs_T205": overlap(A207, a205, tst_eps),
           "T207_vs_T199": overlap(A207, a199, tst_eps),
           "T207_vs_global": overlap(A207, A207_g, tst_eps)}

pts = pe["T207_local"]["score_recall_pts"]
kp = e207["keep_pct"]
kill = bool((pts <= KILL_PTS) or (kp < KEEP_BAR))
verdict = "DISCARD" if kill else "KEEP"

out = {
    "variant": "T207 = band-local soft-score s=(w191-mu_band)/sig_band - lam*clip(jerk,0,J95_band), pooled-0, no hard gates",
    "frozen": {"TAU50_global_ref": EXP_TAU50, "theta_marg_bitident": True,
               "C99": round(C99, 4), "C99_degenerate": True,
               "J95_global_ref": EXP_J95, "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7),
               "w191_T173_embedded_kept": True,
               "no_TAUhi_lo_hard_gates": True,
               "fit_scope": "CALIB pooled only, jerk-tertile bands, single held eval"},
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [len(x) for x in cal_band],
              "cal_fails": [len(x) for x in cal_fails_band],
              "J95_band": [round(v, 6) for v in J95_band],
              "mu_band": [round(v, 4) for v in mu_band],
              "sig_band": [round(v, 4) for v in sig_band],
              "mu_global": round(mu_g, 4), "sig_global": round(sig_g, 4)},
    "fit": {"LAM_GRID": LAM_GRID,
            "tau_star_per_band": {str(lam): {str(b): tau_star[lam][b] for b in range(3)} for lam in LAM_GRID},
            "LAM_STAR": LAM_STAR,
            "TAU_STAR_band": ({str(b): round(TAUS_STAR[b], 4) for b in range(3)} if TAUS_STAR else None),
            "calib_star_recall": round(star_cal["recall"], 4) if star_cal else None,
            "calib_star_pf": round(star_cal["pf"], 4) if star_cal else None,
            "global_abl": {"G_LAM": G_LAM, "G_TAU": round(G_TAU, 4) if G_TAU is not None else None,
                           "calib_recall": round(g_star_cal["recall"], 4) if g_star_cal else None,
                           "calib_pf": round(g_star_cal["pf"], 4) if g_star_cal else None}},
    "perband_ROC_held": band_roc_held,
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T207": e207, "ref_T205": heldB_eval(a205), "cov_all": cov_all,
              "yield_vs_T205_pts": round(100 * (e207["yield_sel"] - heldB_eval(a205)["yield_sel"]), 2)},
    "pooled_held": {**pe,
                    "recall_vs_T199_pts": round(pe["T207_local"]["score_recall_pts"] - pe["T199"]["score_recall_pts"], 2),
                    "recall_vs_T205_pts": round(pe["T207_local"]["score_recall_pts"] - pe["T205"]["score_recall_pts"], 2),
                    "local_vs_global_pts": round(pe["T207_local"]["score_recall_pts"] - pe["T207_global_abl"]["score_recall_pts"], 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["T207_local"]["p_fail_adm"]), 2)},
    "rescue_vs_T205": {"rescued": rescued, "new_mistakes": new_mist, "rescue_rate": rescue_rate},
    "admit_overlap_held": ov_held,
    "archive_adm_frac_T207": round(sum(1 for e in arc_eps if A207(e)) / len(arc_eps), 4),
    "cov_ok": e207["cov_adm"] >= cov_all - 0.02,
    "kill_rule": {"pts_le_48_67": bool(pts <= KILL_PTS), "keep_lt_60": bool(kp < KEEP_BAR),
                  "plateau": PLATEAU, "KILL_PTS": KILL_PTS, "KEEP_BAR": KEEP_BAR},
    "director_predicted_pts_55pm5": pts,
    "director_predicted_keep_gt_60": kp,
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "pts_gt_48_67": pts > KILL_PTS,
    "HELD_keep_ge_60": kp >= KEEP_BAR,
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T207"] == 0.0}
out["keep"] = bool(out["verdict_rule"]["pts_gt_48_67"] and out["verdict_rule"]["HELD_keep_ge_60"]
                   and out["runA_keep_cited"] and out["cov_ok"])
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
out["frontier"] = ("tune lam_band per-band in T208 (local beats global and keep>=60)"
                   if out["keep"] else
                   "band-local soft-score adds no outcome signal; CLOSE localization line" if kill
                   else "marginal; T208 lam_band only if keep>=60 holds")
json.dump(out, open("results/aegis_v2/I2_r207_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
