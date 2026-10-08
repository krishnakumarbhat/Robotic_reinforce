"""Run 211: T204 = robust global soft-score with clipped jerk (director iter 29).

Director iter 29 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: soft-penalty scoring (T201 line: 93.33 >> T202/T203 46.67).
Diagnosis: hard vetoes destroy signal (T202 vacuous JMAX, T203 bins add 0.0);
  T201 raw-jerk soft score wins but is outlier-sensitive (lam=8000 grid-edge,
  joint lam+TAU in-train CV overfits: fold-mode lam=0 vs tie-break 8000).
Rule: s204(e;lam) = w191(e) - lam*clip(jerk(e),0,J95),
  J95 = Q0.95 over CALIB pooled jerk (train-locked); admit iff s>=TAU_global.
Variation vs T201: raw jerk -> clipped jerk (tail veto capped at lam*J95;
  fixes outlier sensitivity); joint lam+TAU CV -> DECOUPLED fit (lam swept on
  pooled-1 CALIB at fixed Q70-fail TAU rule, then TAU_global locked for the
  70%-fail-recall target yield; no tertile bins — T203 overfit).
Fit (train-locked, single held eval): for each lam in T201 grid,
  TAU70(lam) = Q0.70 of s_lam over CALIB fails (n=17); LAM_STAR = min calib
  P(fail|admit), tie-break smallest lam. Zero per-tool tables; zero rig edits.
Validation: same pooled-1 CALIB scope as T202/T203 + lam=0 ablation (same Q70
  rule, proves jerk adds) + T201 recompute (raw jerk, lam=8000/TAU=75.8) for
  the CALIB-pooled gain check + T199 hard-gate ref.
Verdict (director): DISCARD iff no gain over T201 on CALIB pooled-1
  (calib recall <= T201 AND calib pf >= T201) -> kill soft-penalty frontier;
  KEEP iff HELD keep>=70 AND pts>=70 AND calib gain AND no cov regression.
Predicted 100-110pts, P(keep>=70) high. Frozen I7+I10. G7-clean, seg 15."""

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
EXP_T201_LAM = 8000
EXP_T201_TAU = 75.8
LAM_GRID = [0, 100, 250, 500, 1000, 2000, 4000, 8000]


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
assert tst_hdr["pose_noise_cfg"] == "0.01,2" and arc_hdr["pose_noise_cfg"] == "2.0,2"

# Bit-identical recompute (T-series precedent)
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


def t173(e):
    return e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP


def w191(e):
    if not t173(e):
        return 0.0
    return min(1.0 / (max(e["slip_m"], FLOOR) + EPS), C99)


calB = [e for e in cal_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
calB_w = sorted(w191(e) for e in calB if t173(e))
assert abs(st.median(calB_w) - EXP_TAU50) < 5e-5, st.median(calB_w)
assert len(calB_w) == 16, len(calB_w)

cal_fails = [e for e in cal_eps if not e["success"]]
assert len(cal_fails) == 17, len(cal_fails)

# ---- J95 clip point, CALIB-pooled train-locked ----
J95 = qceil([e["jerk"] for e in cal_eps], 0.95)
JMAX_CAL = max(e["jerk"] for e in cal_eps)
tail_frac_cal = sum(1 for e in cal_eps if e["jerk"] > J95) / len(cal_eps)
tail_frac_held = sum(1 for e in tst_eps if e["jerk"] > J95) / len(tst_eps)


def jc(e):
    return min(max(e["jerk"], 0.0), J95)


def s_clip(e, lam):
    return w191(e) - lam * jc(e)


def s_raw(e, lam):
    return w191(e) - lam * e["jerk"]


# ---- decoupled fit on pooled-1 CALIB: lam sweep at fixed Q70-fail TAU rule ----
calib_grid = {}
for lam in LAM_GRID:
    tau70 = qceil([s_clip(e, lam) for e in cal_fails], 0.70)
    adm = [e for e in cal_eps if s_clip(e, lam) >= tau70]
    na = len(adm)
    pf = sum(1 for e in adm if not e["success"]) / na if na else 1.0
    caught = sum(1 for e in cal_fails if s_clip(e, lam) < tau70)
    calib_grid[lam] = {"TAU70": tau70, "n_adm": na,
                       "p_fail_adm": round(pf, 4),
                       "recall": round(caught / len(cal_fails), 4)}
best_pf = min(v["p_fail_adm"] for v in calib_grid.values())
cands = [lam for lam in LAM_GRID if calib_grid[lam]["p_fail_adm"] == best_pf]
LAM_STAR = min(cands)
TAU_STAR = calib_grid[LAM_STAR]["TAU70"]
edge_flag = LAM_STAR in (LAM_GRID[0], LAM_GRID[-1])

a204 = lambda e: s_clip(e, LAM_STAR) >= TAU_STAR  # noqa: E731
a199 = lambda e: w191(e) >= EXP_TAU50  # noqa: E731
TAU_LAM0 = qceil([s_clip(e, 0) for e in cal_fails], 0.70)
a204_lam0 = lambda e: s_clip(e, 0) >= TAU_LAM0  # noqa: E731
a201 = lambda e: s_raw(e, EXP_T201_LAM) >= EXP_T201_TAU  # noqa: E731

# T201-equivalent on CALIB pooled-1 (gain check, in-train scope for both)
cal_adm201 = [e for e in cal_eps if a201(e)]
cal_pf201 = sum(1 for e in cal_adm201 if not e["success"]) / len(cal_adm201)
cal_rec201 = sum(1 for e in cal_fails if not a201(e)) / len(cal_fails)
cal_adm204 = [e for e in cal_eps if a204(e)]
cal_pf204 = sum(1 for e in cal_adm204 if not e["success"]) / len(cal_adm204)
cal_rec204 = sum(1 for e in cal_fails if not a204(e)) / len(cal_fails)
calib_gain = (cal_rec204 > cal_rec201) or (
    cal_rec204 == cal_rec201 and cal_pf204 < cal_pf201)

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
            "score_sel_pts": round(100 * s / nn, 2) if nn else 100.0,
            "cov_adm": round(st.mean(e["coverage_cont"] for e in adm), 4) if adm else 0.0}


def pooled_eval(fn):
    adm = [e for e in tst_eps if fn(e)]
    fails = [e for e in tst_eps if not e["success"]]
    caught = sum(1 for e in fails if not fn(e))
    nn = len(adm)
    pf = sum(1 for e in adm if not e["success"]) / nn if nn else 0.0
    return {"n_adm": nn, "p_fail_adm": round(pf, 4),
            "recall_fail_reject": round(caught / len(fails), 4),
            "score_recall_pts": round(100 * caught / len(fails), 2)}


e204 = heldB_eval(a204)
e199 = heldB_eval(a199)
e201 = heldB_eval(a201)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
p204 = pooled_eval(a204)
p199 = pooled_eval(a199)
p201 = pooled_eval(a201)
p_lam0 = pooled_eval(a204_lam0)

succ_rej199 = [e for e in tst_eps if e["success"] and not a199(e)]
rescued = [e for e in succ_rej199 if a204(e)]
rescue_rate = round(len(rescued) / len(succ_rej199), 4) if succ_rej199 else 0.0
new_mistakes = [e for e in tst_eps if not e["success"] and a204(e) and not a199(e)]
hB_rej199_succ = [e for e in hB if e["success"] and not a199(e)]
hB_rescued = [e for e in hB_rej199_succ if a204(e)]

out = {
    "variant": "T204 = robust global soft-score: s=w191-lam*clip(jerk,0,J95), admit iff s>=TAU_global (decoupled lam sweep + Q70-fail TAU, zero bins)",
    "frozen": {"TAU50_global_ref": EXP_TAU50, "theta_marg_bitident": True,
               "C99": round(C99, 4), "C99_degenerate": True,
               "calB_adm_n": len(calB_w), "cal_fails_n": len(cal_fails),
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7),
               "fit_scope": "pooled-1 CALIB lam sweep at fixed Q70-fail TAU rule (decoupled, single held eval)"},
    "clip": {"J95_calib_pooled": round(J95, 6), "JMAX_calib": round(JMAX_CAL, 6),
             "tail_frac_jerk_gt_J95_calib": round(tail_frac_cal, 4),
             "tail_frac_jerk_gt_J95_held": round(tail_frac_held, 4),
             "max_penalty_avoided_at_lam_star": round(LAM_STAR * (JMAX_CAL - J95), 4)},
    "fit_calib_grid": {str(lam): v for lam, v in calib_grid.items()},
    "locked": {"LAM_STAR": LAM_STAR, "TAU_STAR": round(TAU_STAR, 4),
               "TAU_LAM0_same_rule": round(TAU_LAM0, 4),
               "lam_at_grid_edge": bool(edge_flag)},
    "calib_pooled_gain_vs_T201": {
        "T204_recall": round(cal_rec204, 4), "T204_p_fail_adm": round(cal_pf204, 4),
        "T201_recall": round(cal_rec201, 4), "T201_p_fail_adm": round(cal_pf201, 4),
        "gain": bool(calib_gain)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T204": e204, "ref_T199": e199, "abl_T201": e201, "cov_all": cov_all,
              "yield_vs_T199_pts": round(100 * (e204["yield_sel"] - e199["yield_sel"]), 2),
              "yield_vs_T201_pts": round(100 * (e204["yield_sel"] - e201["yield_sel"]), 2)},
    "pooled_held": {"T173": p_t173, "T204": p204, "ref_T199": p199,
                    "abl_T201": p201, "abl_lam0_same_rule": p_lam0,
                    "jerk_adds_vs_lam0_pts":
                        round(p204["score_recall_pts"] - p_lam0["score_recall_pts"], 2),
                    "recall_vs_T199_pts":
                        round(p204["score_recall_pts"] - p199["score_recall_pts"], 2),
                    "recall_vs_T201_pts":
                        round(p204["score_recall_pts"] - p201["score_recall_pts"], 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - p204["p_fail_adm"]), 2)},
    "rescue_vs_T199": {"held_succ_rejected_by_T199": len(succ_rej199),
                       "rescued_by_T204": len(rescued),
                       "rescue_rate": rescue_rate,
                       "new_mistakes_fails_adm_by_T204_not_T199": len(new_mistakes),
                       "heldB_succ_rejected_by_T199": len(hB_rej199_succ),
                       "heldB_rescued": len(hB_rescued),
                       "precision_no_drop_pooled": bool(p204["p_fail_adm"] <= p199["p_fail_adm"])},
    "archive_adm_frac_T204": round(sum(1 for e in arc_eps if a204(e)) / len(arc_eps), 4),
    "cov_ok": e204["cov_adm"] >= cov_all - 0.02,
    "director_predicted_pts_100_110": p204["score_recall_pts"],
    "director_predicted_keep_ge_70": e204["keep_pct"],
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "calib_gain_vs_T201": bool(calib_gain),
    "HELD_keep_ge_70": e204["keep_pct"] >= 70.0,
    "pts_ge_70": p204["score_recall_pts"] >= 70.0,
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T204"] == 0.0}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r204_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
