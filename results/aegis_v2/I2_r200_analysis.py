"""Run 207: T200 = global-only linear soft score (director iter 25).

Director iter 25 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: global-only soft gate, zero per-tool tables until global >60.
Proposal T200 (pre-reg, single eval): replace OR/AND union with linear score
  s(e;lam) = w191(e) - lam*jerk(e), Admit200 iff s >= TAU_global
  w191 = min(1[T173]/(max(slip,0.005)+0.005),C99) T191 joint weight (bit-identical
  recompute + assert before eval); T173 embedded via w=0 (no extra THETA check).
Variation vs T197/T198/T199: 1 lam + 1 TAU (2 params) vs 2xT tables + TxTHETA;
  tests whether a soft jerk tradeoff rescues successes the T199 hard gate vetoes,
  without per-tool overfit. (Director cites "T205 hard-gate loss (40.0)" — no T205
  exists; read as T198 40.0, the worst hard-gate pooled recall in the line.)
FIT (one documented deviation): director says "grid lam on held-out"; this run
  grids lam on CALIB (train-locked, T-series precedent — fitting on held would be
  test-peeking) and evaluates ONCE on held. TAU_global = Q70 of s over calib
  FAIL episodes (70% failure-recall target, not conformal 30/50 admitted
  quantiles). lam selected on calib at fixed ~70% calib recall by min
  P(fail|admit) (precision), tie-break smallest lam. lam=0 sanity arm must
  reproduce the T199-class hard gate shape (s=w191).
Validation: same split/scorer as T199 (frozen R173 calib/test/archive; pooled
  failure-reject recall x100 = pts; heldB yield = n_adm_success/20).
KEEP iff: runA keep cited AND held pooled recall >=70 AND heldB precision==1.0
  (no precision drop vs T199) AND rescue-rate>0 (admits >=1 held success T199
  rejects) AND no cov regression AND archive adm 0. Else DISCARD.
Predicted 71.0 KEEP (director: +6.7 jerk-gate + ~18 soft-tradeoff recall).
Frozen I7+I10; zero rig edits; no rig run; no refit. G7-clean, seg 15."""

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
EXP_TAU30 = 55.9211
EXP_THETA_MARG = 0.014133
LAM_GRID = [0, 250, 500, 1000, 2000, 4000]


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
TAU50 = st.median(calB_w)
TAU30 = qceil(calB_w, 0.30)
assert abs(TAU50 - EXP_TAU50) < 5e-5, TAU50
assert abs(TAU30 - EXP_TAU30) < 5e-5, TAU30
assert len(calB_w) == 16, len(calB_w)

a199 = lambda e: w191(e) >= TAU50  # T199 ref (global hard gate)


def s_of(e, lam):
    return w191(e) - lam * e["jerk"]


# ---- train-locked fit on CALIB: TAU70(lam) = Q70 of s over calib FAILS ----
cal_fails = [e for e in cal_eps if not e["success"]]
assert len(cal_fails) == 17, len(cal_fails)
calib_grid = {}
for lam in LAM_GRID:
    tau70 = qceil([s_of(e, lam) for e in cal_fails], 0.70)
    adm = lambda e, _l=lam, _t=tau70: s_of(e, _l) >= _t
    ca = [e for e in cal_eps if adm(e)]
    n_adm = len(ca)
    pf = sum(1 for e in ca if not e["success"]) / n_adm if n_adm else 0.0
    caught = sum(1 for e in cal_fails if not adm(e))
    calib_grid[lam] = {"TAU70": tau70, "n_adm": n_adm,
                       "p_fail_adm": round(pf, 4),
                       "recall": round(caught / len(cal_fails), 4)}
# select: min calib P(fail|admit), tie-break smallest lam
best_pf = min(v["p_fail_adm"] for v in calib_grid.values())
cands = [lam for lam in LAM_GRID if calib_grid[lam]["p_fail_adm"] == best_pf]
LAM_STAR = min(cands)
TAU_STAR = calib_grid[LAM_STAR]["TAU70"]
adm200 = lambda e: s_of(e, LAM_STAR) >= TAU_STAR

# lam=0 sanity: shape check vs T199-class hard gate on held (TAU differs by
# construction — Q70-fail vs median-admitted — so report, not assert equal)
a200_lam0 = lambda e: s_of(e, 0) >= qceil([s_of(x, 0) for x in cal_fails], 0.70)

hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
assert len(hB) == 20
raw_keep = sum(1 for e in hB if e["success"]) / 20.0


def heldB_eval(adm_fn):
    adm = [e for e in hB if adm_fn(e)]
    s = sum(1 for e in adm if e["success"])
    n_adm = len(adm)
    return {"n_adm": n_adm, "yield_sel": round(s / 20.0, 4),
            "keep_pct": round(100 * n_adm / 20.0, 2),
            "precision_adm": round(s / n_adm, 4) if n_adm else 1.0,
            "score_sel_pts": round(100 * s / n_adm, 2) if n_adm else 100.0,
            "cov_adm": round(st.mean(e["coverage_cont"] for e in adm), 4) if adm else 0.0}


e200 = heldB_eval(adm200)
e199 = heldB_eval(a199)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)


def pooled_eval(adm_fn):
    adm = [e for e in tst_eps if adm_fn(e)]
    fails = [e for e in tst_eps if not e["success"]]
    caught = sum(1 for e in fails if not adm_fn(e))
    n_adm = len(adm)
    pf = sum(1 for e in adm if not e["success"]) / n_adm if n_adm else 0.0
    return {"n_adm": n_adm, "p_fail_adm": round(pf, 4),
            "recall_fail_reject": round(caught / len(fails), 4),
            "score_recall_pts": round(100 * caught / len(fails), 2)}


p_t173 = pooled_eval(t173)
p200 = pooled_eval(adm200)
p199 = pooled_eval(a199)
p200_lam0 = pooled_eval(a200_lam0)

# rescue analysis vs T199 (pooled held + heldB)
succ_rej199 = [e for e in tst_eps if e["success"] and not a199(e)]
rescued = [e for e in succ_rej199 if adm200(e)]
rescue_rate = round(len(rescued) / len(succ_rej199), 4) if succ_rej199 else 0.0
new_mistakes = [e for e in tst_eps
                if not e["success"] and adm200(e) and not a199(e)]
hB_rej199_succ = [e for e in hB if e["success"] and not a199(e)]
hB_rescued = [e for e in hB_rej199_succ if adm200(e)]

out = {
    "variant": "T200 = global-only linear soft score: s=w191-lam*jerk, admit iff s>=TAU_global (2 params, zero per-tool tables, zero buffer/shrinkage)",
    "frozen": {"TAU50_global_ref": round(TAU50, 4), "TAU30_ref": round(TAU30, 4),
               "theta_marg_bitident": True, "C99": round(C99, 4),
               "C99_degenerate": True, "calB_adm_n": len(calB_w),
               "cal_fails_n": len(cal_fails),
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7),
               "fit_deviation": "lam gridded on CALIB train-locked (not held-out per director FIT line: held-fit would be test-peeking); single held eval"},
    "fit_calib_grid": {str(lam): v for lam, v in calib_grid.items()},
    "locked": {"LAM_STAR": LAM_STAR, "TAU_STAR": round(TAU_STAR, 4)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T200": e200, "ref_T199": e199, "cov_all": cov_all,
              "yield_vs_T199_pts": round(100 * (e200["yield_sel"] - e199["yield_sel"]), 2)},
    "pooled_held": {"T173": p_t173, "T200": p200, "ref_T199": p199,
                    "ref_lam0_Q70fail": p200_lam0,
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - p200["p_fail_adm"]), 2),
                    "recall_vs_T199_pts":
                        round(p200["score_recall_pts"] - p199["score_recall_pts"], 2)},
    "rescue_vs_T199": {"held_succ_rejected_by_T199": len(succ_rej199),
                       "rescued_by_T200": len(rescued),
                       "rescue_rate": rescue_rate,
                       "new_mistakes_fails_adm_by_T200_not_T199": len(new_mistakes),
                       "heldB_succ_rejected_by_T199": len(hB_rej199_succ),
                       "heldB_rescued": len(hB_rescued),
                       "precision_no_drop_pooled": bool(p200["p_fail_adm"] <= p199["p_fail_adm"])},
    "archive_adm_frac_T200": round(sum(1 for e in arc_eps if adm200(e)) / len(arc_eps), 4),
    "cov_ok": e200["cov_adm"] >= cov_all - 0.02,
    "director_predicted_71": p200["score_recall_pts"],
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "recall_ge_70": p200["score_recall_pts"] >= 70.0,
    "heldB_precision_1": e200["precision_adm"] >= 1.0,
    "rescue_rate_gt_0": rescue_rate > 0,
    "precision_no_drop_vs_T199": out["rescue_vs_T199"]["precision_no_drop_pooled"],
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T200"] == 0.0}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r200_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
