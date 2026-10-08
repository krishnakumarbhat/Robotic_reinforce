"""Run 215: T208 = shrunk band-local + global soft-score, pooled-1, no hard gates (director iter 33).

Director iter 33 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: keep T207 band-local soft-score only climber 46.67->60.0.
Variation vs T207: s = z_shrunk + z_global - jerk_pen, no hard gates, pooled-1 CALIB.
  z_shrunk=(w191-mu_shrunk)/sig_pooled, mu_shrunk shrunk to global by band-n:
    mu_shrunk_b=(n_b*mu_b+n0*mu_g)/(n_b+n0), n0=20 train-locked (n_b=40 -> w=2/3 band).
  sig_pooled=sig_g (global CALIB stdev, pooled-1).
  z_global=(w191-mu_g)/sig_g.
  jerk_pen=lam*clip(jerk,0,J95_global)/J95_global, J95_global=0.015175, global not band-local.
Fit (train-locked, CALIB pooled only, single held eval):
  bands=CALIB jerk tertiles; lam grid {0,0.25,0.5,1.0}; global TAU over CALIB fail-s
  quantiles Q{0.5..0.9} (qceil, deduped); pick recall>=0.65 then min pf, loosest tie-break;
  LAM* by pooled recall>=0.65 then min pf, smallest-lam tie-break.
Validation (all train-locked, single held eval): ablate lam=0 @same TAU rule;
  raw-mu (mu_band/sig_band, same lam*) vs shrunk; pooled-1 (global TAU) vs pooled-0
  (per-band TAU on same s, T207-style fit).
Kill: DISCARD iff pooled recall pts<=48.67 OR heldB keep<60 OR shrunk<=raw (director fail).
Predicted 71.7pts KEEP. Frozen I7+I10. Zero rig edits, frozen R173 files. G7-clean, seg 15."""

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
LAM_GRID = [0, 0.25, 0.5, 1.0]
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
PLATEAU = 46.67
KILL_PTS = PLATEAU + 2.0
KEEP_BAR = 60.0
N0 = 20


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
cal_fails = [e for e in cal_eps if not e["success"]]

mu_band, sig_band = [], []
for b in range(3):
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

mu_g = st.mean([w191(e) for e in cal_eps])
try:
    sig_g = st.stdev([w191(e) for e in cal_eps])
except Exception:
    sig_g = 1.0
if not (sig_g > 1e-9):
    sig_g = 1.0
SIG_P = sig_g
J95 = EXP_J95
n_b = [len(cal_band[b]) for b in range(3)]
mu_shrunk = [(n_b[b] * mu_band[b] + N0 * mu_g) / (n_b[b] + N0) for b in range(3)]


def clip_j(e):
    return min(max(e["jerk"], 0.0), J95)


def s_t208(e, lam):
    b = band_of(e)
    w = w191(e)
    z_shrunk = (w - mu_shrunk[b]) / SIG_P
    z_global = (w - mu_g) / sig_g
    return z_shrunk + z_global - lam * clip_j(e) / J95


def s_raw(e, lam):
    b = band_of(e)
    w = w191(e)
    z_raw = (w - mu_band[b]) / sig_band[b]
    z_global = (w - mu_g) / sig_g
    return z_raw + z_global - lam * clip_j(e) / J95


def pooled_stats(fn, eps):
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if fn(e)]
    caught = sum(1 for e in fails if not fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


def fit_global(score_fn, lam):
    sc_fail = sorted(score_fn(e, lam) for e in cal_fails)
    vals = []
    for q in Q_GRID:
        v = qceil(sc_fail, q)
        if not vals or abs(v - vals[-1]) > 1e-12:
            vals.append(v)
    best = None
    for tv in vals:
        s = pooled_stats(lambda e, lam=lam, tv=tv: score_fn(e, lam) >= tv, cal_eps)
        if s["recall"] >= 0.65:
            key = (s["pf"], tv)
            if best is None or key < best[0]:
                best = (key, tv, s)
    return vals, ({"TAU": best[1], "recall": best[2]["recall"], "pf": best[2]["pf"],
                   "feasible": True} if best else {"TAU": None, "feasible": False})


lam_pick = {}
for lam in LAM_GRID:
    _, p = fit_global(s_t208, lam)
    lam_pick[lam] = p
feas = [(lam_pick[lam]["pf"], lam) for lam in LAM_GRID
        if lam_pick[lam]["feasible"] and lam_pick[lam]["recall"] >= 0.65]
feas.sort(key=lambda r: (r[0], r[1]))
LAM_STAR = feas[0][1] if feas else None
TAU_STAR = lam_pick[LAM_STAR]["TAU"] if LAM_STAR is not None else None

# pooled-0 ablation: per-band TAU on same s_t208 @LAM_STAR (T207-style)
tau0 = {}
if LAM_STAR is not None:
    for b in range(3):
        fb = [e for e in cal_band[b] if not e["success"]]
        sc = sorted(s_t208(e, LAM_STAR) for e in fb)
        vals = []
        for q in Q_GRID:
            v = qceil(sc, q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        best = None
        for tv in vals:
            ab = [e for e in cal_band[b] if s_t208(e, LAM_STAR) >= tv]
            fails_b = fb
            caught = sum(1 for e in fails_b if not (s_t208(e, LAM_STAR) >= tv))
            pf = sum(1 for e in ab if not e["success"]) / len(ab) if ab else 1.0
            rec = caught / len(fails_b)
            if rec >= 0.65:
                key = (pf, tv)
                if best is None or key < best[0]:
                    best = (key, tv, rec, pf)
        tau0[b] = {"TAU": best[1] if best else None, "feasible": best is not None,
                   "recall": best[2] if best else None, "pf": best[3] if best else None}

A = (lambda e: s_t208(e, LAM_STAR) >= TAU_STAR) if LAM_STAR is not None else (lambda e: False)
_, p_lam0 = fit_global(s_t208, 0)
A_lam0 = (lambda e: s_t208(e, 0) >= p_lam0["TAU"]) if p_lam0["feasible"] else (lambda e: False)
_, p_raw = fit_global(s_raw, LAM_STAR if LAM_STAR is not None else 0.5)
A_raw = (lambda e: s_raw(e, LAM_STAR) >= p_raw["TAU"]) if (LAM_STAR is not None and p_raw["feasible"]) else (lambda e: False)
if LAM_STAR is not None and all(tau0[b]["feasible"] for b in range(3)):
    _t = {b: tau0[b]["TAU"] for b in range(3)}
    A_p0 = (lambda e, _t=_t: s_t208(e, LAM_STAR) >= _t[band_of(e)])
else:
    A_p0 = (lambda e: False)
a205 = lambda e: w191(e) >= T205_TAU  # noqa: E731
a199 = lambda e: w191(e) >= EXP_TAU50  # noqa: E731

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


eB = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = {"T199": pooled_eval(a199), "T205": pooled_eval(a205),
      "T207_local_ref": {"n_adm": 43, "score_recall_pts": 60.0},
      "T208": pooled_eval(A), "T208_lam0": pooled_eval(A_lam0),
      "T208_rawmu": pooled_eval(A_raw), "T208_pooled0": pooled_eval(A_p0)}
pts = pe["T208"]["score_recall_pts"]
kp = eB["keep_pct"]
raw_pts = pe["T208_rawmu"]["score_recall_pts"]
shrunk_beats_raw = bool(pts > raw_pts)
kill = bool((pts <= KILL_PTS) or (kp < KEEP_BAR) or (not shrunk_beats_raw))

out = {
    "variant": "T208 = s=z_shrunk+z_global-jerk_pen, pooled-1, no hard gates",
    "frozen": {"TAU50_global_ref": EXP_TAU50, "theta_marg_bitident": True, "C99": round(C99, 4),
               "J95_global": J95, "fit_scope": "CALIB pooled only, jerk-tertile bands, single held eval",
               "w191_T173_embedded_kept": True, "no_TAUhi_lo_hard_gates": True},
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "mu_band": [round(v, 4) for v in mu_band],
              "sig_band": [round(v, 4) for v in sig_band],
              "mu_global": round(mu_g, 4), "sig_global": round(sig_g, 4),
              "sig_pooled": round(SIG_P, 4),
              "mu_shrunk": [round(v, 4) for v in mu_shrunk], "shrink_n0": N0,
              "shrink_w_band": round(n_b[0] / (n_b[0] + N0), 4)},
    "fit": {"LAM_GRID": LAM_GRID,
            "lam_pick": {str(l): {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in d.items()} for l, d in lam_pick.items()},
            "LAM_STAR": LAM_STAR,
            "TAU_STAR": round(TAU_STAR, 4) if TAU_STAR is not None else None,
            "calib_star_recall": round(lam_pick[LAM_STAR]["recall"], 4) if LAM_STAR is not None else None,
            "calib_star_pf": round(lam_pick[LAM_STAR]["pf"], 4) if LAM_STAR is not None else None,
            "lam0_TAU": round(p_lam0["TAU"], 4) if p_lam0["feasible"] else None,
            "rawmu_TAU": round(p_raw["TAU"], 4) if p_raw["feasible"] else None,
            "pooled0_TAU_band": {str(b): (round(tau0[b]["TAU"], 4) if tau0[b]["TAU"] is not None else None) for b in range(3)}},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T208": eB, "ref_T205": heldB_eval(a205), "cov_all": cov_all,
              "yield_vs_T205_pts": round(100 * (eB["yield_sel"] - heldB_eval(a205)["yield_sel"]), 2)},
    "pooled_held": {**pe,
                    "recall_vs_T199_pts": round(pe["T208"]["score_recall_pts"] - pe["T199"]["score_recall_pts"], 2),
                    "recall_vs_T205_pts": round(pe["T208"]["score_recall_pts"] - pe["T205"]["score_recall_pts"], 2),
                    "recall_vs_T207_pts": round(pe["T208"]["score_recall_pts"] - 60.0, 2),
                    "shrunk_vs_raw_pts": round(pe["T208"]["score_recall_pts"] - pe["T208_rawmu"]["score_recall_pts"], 2),
                    "lam_vs_lam0_pts": round(pe["T208"]["score_recall_pts"] - pe["T208_lam0"]["score_recall_pts"], 2),
                    "pooled1_vs_pooled0_pts": round(pe["T208"]["score_recall_pts"] - pe["T208_pooled0"]["score_recall_pts"], 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["T208"]["p_fail_adm"]), 2)},
    "archive_adm_frac_T208": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": eB["cov_adm"] >= cov_all - 0.02,
    "kill_rule": {"pts_le_48_67": bool(pts <= KILL_PTS), "keep_lt_60": bool(kp < KEEP_BAR),
                  "shrunk_le_raw": bool(not shrunk_beats_raw),
                  "plateau": PLATEAU, "KILL_PTS": KILL_PTS, "KEEP_BAR": KEEP_BAR},
    "director_predicted_pts_71_7": pts,
    "director_predicted_keep": kp,
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "pts_gt_48_67": pts > KILL_PTS,
    "HELD_keep_ge_60": kp >= KEEP_BAR,
    "shrunk_gt_raw": shrunk_beats_raw,
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T208"] == 0.0}
out["keep"] = bool(out["verdict_rule"]["pts_gt_48_67"] and out["verdict_rule"]["HELD_keep_ge_60"]
                   and out["verdict_rule"]["shrunk_gt_raw"] and out["runA_keep_cited"] and out["cov_ok"])
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r208_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
