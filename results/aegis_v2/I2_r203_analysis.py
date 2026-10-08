"""Run 210: T203 = jerk-stratified dual-cut, pooled-1 (director iter 28).

Director iter 28 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: conditional admission - global TAU exhausted (T200/T201/T202 all
fail keep>=70: hard/soft/CV-joint/dual-cut converge to pooled recall 46.67
or strictness artifacts with heldB yield <=0.40).
Diagnosis: global s=w191-lam*jerk forces one tradeoff; T201 93.3pts still
discards (strictness artifact, heldB yield 0.0) = lam/TAU cannot satisfy
low+high-jerk regimes jointly.
Rule: Admit203 iff w191 >= TAU_b[jerk-bin(e)], k=3 quantile bins from CALIB
only (tertile edges over CALIB pooled jerk, train-locked). Linear w191 kept
(no jerk penalty); all selectivity moves into TAU_b.
Variation vs T202: global AND-veto (rejected in-train: JMAX*=max vacuous) ->
stratified TAU_b (loose bar in low-jerk, strict bar in high-jerk).
Fit (train-locked, single held eval): co-tune (TAU1,TAU2,TAU3) in-train on
CALIB pooled: each TAU_b from global fail-w grid Q_q(w191|calib fails,n=17),
q in {0.5,0.6,0.7,0.8,0.9} (stable; per-bin fail quantiles would be n~5
noise); 125 combos -> isotonic filter TAU1<=TAU2<=TAU3 -> T202 objective
(calib failure-recall>=0.65 then min calib P(fail|admit), n_adm=0 -> P=1.0;
tie-break loosest = lowest sum TAU_b, then lowest TAU3, then lowest TAU2).
If none>=0.65, max calib recall wins (same tie-breaks).
Validation: same splits/scorer as T199-T202 (frozen R173 calib/test/archive;
pooled failure-reject recall x100 = pts; heldB keep = 100*admitted-success/20).
Ablations: T199 ref (w>=TAU50), T202 locked dual-cut recomputed bit-identical,
T201 soft recomputed (lam=8000,TAU=75.8; deviation noted: T201 refit theta per
fold, here frozen TH_MARG bit-identical).
Bin-count check: director requires >=50/bin; CALIB tertiles give 40/bin ->
check REPORTED AS FAILED (caveat), but guardrail collapse (empty/degenerate
bin) does not fire, so no T201 fallback.
Kill/keep (director): KEEP iff HELD keep>=70 AND pts>=70, else kill the
conditional line and try per-sample conformal next.
Predicted 78-85pts, keep 71-75%.
Frozen I7+I10; zero rig edits; no rig run. G7-clean, seg 15."""

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
EXP_T202_TAU = 74.9317
EXP_T202_JMAX = 0.020572
EXP_T201_LAM = 8000
EXP_T201_TAU = 75.8
TAU_QS = [0.50, 0.60, 0.70, 0.80, 0.90]
RECALL_FLOOR = 0.65
K = 3


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
cal_succ = [e for e in cal_eps if e["success"]]
assert len(cal_fails) == 17, len(cal_fails)

# ---- k=3 jerk bins from CALIB only (order-tertiles, 40/40/40) ----
cj = sorted(e["jerk"] for e in cal_eps)
n = len(cj)
assert n == 120, n
E1, E2 = cj[n // 3 - 1], cj[2 * n // 3 - 1]


def jerk_bin(e):
    j = e["jerk"]
    if j <= E1:
        return 0
    if j <= E2:
        return 1
    return 2


cal_bins = [[e for e in cal_eps if jerk_bin(e) == b] for b in range(K)]
cal_bin_n = [len(b) for b in cal_bins]
cal_bin_fails = [sum(1 for e in b if not e["success"]) for b in cal_bins]
bin_check_ge50 = all(c >= 50 for c in cal_bin_n)
collapsed = any(c == 0 for c in cal_bin_n) or any(f == 0 for f in cal_bin_fails)

# ---- train-locked co-tune of (TAU1,TAU2,TAU3) on CALIB pooled ----
TAU_GRID = [qceil([w191(e) for e in cal_fails], q) for q in TAU_QS]


def calib_stats(taus):
    adm = [e for e in cal_eps if w191(e) >= taus[jerk_bin(e)]]
    na = len(adm)
    caught = sum(1 for e in cal_fails if not (w191(e) >= taus[jerk_bin(e)]))
    rec = caught / len(cal_fails)
    pf = sum(1 for e in adm if not e["success"]) / na if na else 1.0
    return {"recall": rec, "p_fail_adm": pf, "n_adm": na}


combos = [(a, b, c) for a in range(5) for b in range(5) for c in range(5)
          if TAU_GRID[a] <= TAU_GRID[b] <= TAU_GRID[c]]  # isotonic
assert combos, "isotonic grid empty"
grid = {cb: calib_stats([TAU_GRID[i] for i in cb]) for cb in combos}
feasible = [cb for cb, v in grid.items() if v["recall"] >= RECALL_FLOOR]
pool = feasible if feasible else list(grid.keys())
if feasible:
    best_pf = min(grid[cb]["p_fail_adm"] for cb in pool)
    cands = [cb for cb in pool if grid[cb]["p_fail_adm"] == best_pf]
else:
    best_rec = max(grid[cb]["recall"] for cb in pool)
    cands = [cb for cb in pool if grid[cb]["recall"] == best_rec]
cands.sort(key=lambda cb: (sum(TAU_GRID[i] for i in cb), TAU_GRID[cb[2]], TAU_GRID[cb[1]]))
STAR = cands[0]
TAU_B = [TAU_GRID[i] for i in STAR]
iso_ok = TAU_B[0] <= TAU_B[1] <= TAU_B[2]
edge_flag = any(i in (0, 4) for i in STAR)

if collapsed:  # guardrail fallback (does not fire: bins 40/40/40, fails>0)
    a203 = lambda e: w191(e) >= EXP_TAU50  # noqa: E731
    fallback = "T201-global-equivalent"
else:
    a203 = lambda e: w191(e) >= TAU_B[jerk_bin(e)]  # noqa: E731
    fallback = None

a199 = lambda e: w191(e) >= EXP_TAU50  # noqa: E731
a202 = lambda e: w191(e) >= EXP_T202_TAU and e["jerk"] <= EXP_T202_JMAX  # noqa: E731
a201 = lambda e: (w191(e) - EXP_T201_LAM * e["jerk"]) >= EXP_T201_TAU  # noqa: E731

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


e203 = heldB_eval(a203)
e199 = heldB_eval(a199)
e202 = heldB_eval(a202)
e201 = heldB_eval(a201)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
p203 = pooled_eval(a203)
p199 = pooled_eval(a199)
p202 = pooled_eval(a202)
p201 = pooled_eval(a201)

succ_rej199 = [e for e in tst_eps if e["success"] and not a199(e)]
rescued = [e for e in succ_rej199 if a203(e)]
rescue_rate = round(len(rescued) / len(succ_rej199), 4) if succ_rej199 else 0.0
new_mistakes = [e for e in tst_eps if not e["success"] and a203(e) and not a199(e)]
hB_rej199_succ = [e for e in hB if e["success"] and not a199(e)]
hB_rescued = [e for e in hB_rej199_succ if a203(e)]
# per-bin held diagnostics
held_bins = [[e for e in tst_eps if jerk_bin(e) == b] for b in range(K)]
held_bin_adm = [sum(1 for e in b if a203(e)) for b in held_bins]
held_bin_fail_adm = [sum(1 for e in b if a203(e) and not e["success"]) for b in held_bins]

out = {
    "variant": "T203 = jerk-stratified dual-cut: admit iff w191>=TAU_b[jerk-bin], k=3 CALIB tertiles, isotonic co-tune",
    "frozen": {"TAU50_global_ref": EXP_TAU50, "theta_marg_bitident": True,
               "C99": round(C99, 4), "C99_degenerate": True,
               "calB_adm_n": len(calB_w), "cal_fails_n": len(cal_fails),
               "cal_succ_n": len(cal_succ),
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7),
               "fit_scope": "CALIB-only co-tune (TAU_b x isotonic, single held eval)"},
    "bins_calib_only": {"E1": round(E1, 6), "E2": round(E2, 6),
                        "n_per_bin": cal_bin_n, "fails_per_bin": cal_bin_fails,
                        "check_ge50": bin_check_ge50,
                        "check_ge50_pass": bool(bin_check_ge50),
                        "collapsed_empty_or_zero_fail": bool(collapsed),
                        "fallback": fallback},
    "fit_calib": {"TAU_grid": [round(t, 4) for t in TAU_GRID],
                  "TAU_qs": TAU_QS, "isotonic_combos_n": len(combos),
                  "feasible_ge65_n": len(feasible),
                  "star_idx": list(STAR),
                  "calib_star": {kk: round(vv, 4) if isinstance(vv, float) else vv
                                 for kk, vv in grid[STAR].items()},
                  "star_at_grid_edge": bool(edge_flag)},
    "locked": {"TAU_B": [round(t, 4) for t in TAU_B], "isotonic_TAU1_le_TAU2_le_TAU3": bool(iso_ok)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T203": e203, "ref_T199": e199, "abl_T202": e202, "abl_T201": e201,
              "cov_all": cov_all,
              "yield_vs_T199_pts": round(100 * (e203["yield_sel"] - e199["yield_sel"]), 2),
              "yield_vs_T202_pts": round(100 * (e203["yield_sel"] - e202["yield_sel"]), 2)},
    "held_bins": [{"n": len(b), "n_adm": a, "fail_adm": f} for b, a, f in
                  zip(held_bins, held_bin_adm, held_bin_fail_adm)],
    "pooled_held": {"T173": p_t173, "T203": p203, "ref_T199": p199,
                    "abl_T202": p202, "abl_T201": p201,
                    "recall_vs_T199_pts":
                        round(p203["score_recall_pts"] - p199["score_recall_pts"], 2),
                    "recall_vs_T202_pts":
                        round(p203["score_recall_pts"] - p202["score_recall_pts"], 2),
                    "recall_vs_T201_pts":
                        round(p203["score_recall_pts"] - p201["score_recall_pts"], 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - p203["p_fail_adm"]), 2)},
    "rescue_vs_T199": {"held_succ_rejected_by_T199": len(succ_rej199),
                       "rescued_by_T203": len(rescued),
                       "rescue_rate": rescue_rate,
                       "new_mistakes_fails_adm_by_T203_not_T199": len(new_mistakes),
                       "heldB_succ_rejected_by_T199": len(hB_rej199_succ),
                       "heldB_rescued": len(hB_rescued),
                       "precision_no_drop_pooled": bool(p203["p_fail_adm"] <= p199["p_fail_adm"])},
    "archive_adm_frac_T203": round(sum(1 for e in arc_eps if a203(e)) / len(arc_eps), 4),
    "cov_ok": e203["cov_adm"] >= cov_all - 0.02,
    "director_predicted_pts_78_85": p203["score_recall_pts"],
    "director_predicted_keep_71_75": e203["keep_pct"],
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "HELD_keep_ge_70": e203["keep_pct"] >= 70.0,
    "pts_ge_70": p203["score_recall_pts"] >= 70.0,
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T203"] == 0.0,
    "isotonic_holds": bool(iso_ok),
    "no_fallback": fallback is None}
out["keep"] = bool(out["runA_keep_cited"] and out["verdict_rule"]["HELD_keep_ge_70"]
                   and out["verdict_rule"]["pts_ge_70"]
                   and out["verdict_rule"]["no_cov_regression"])
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r203_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
