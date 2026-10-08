"""Run 209: T202 = global dual-cut (TAU + JMAX veto), pooled-1 (director iter 27).

Director iter 27 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: global-only exhausted linear (T199 hard=46.67, T200 soft=46.67,
  T201 CV-joint=artifact); minimal non-linear step before per-tool.
Rule: Admit202 iff w191>=TAU AND jerk<=JMAX (2 params, global-only, no lam,
  no per-tool tables/buffer/shrinkage). w191 = min(1[T173]/(max(slip,0.005)
  +0.005),C99) T191 joint weight, bit-identical recompute + assert before eval.
Variation vs T201: replace s=w191-lam*jerk with AND-veto (lam over-penalizes
  high-w191/mid-jerk winners; veto keeps them, kills only tail jerks).
Fit (train-locked, single held eval): co-tune (TAU,JMAX) on CALIB only.
  TAU grid = Q_q(w191|calib FAILS, n=17), q in {0.5,0.6,0.7,0.8,0.9};
  JMAX grid = Q_q(jerk|calib SUCCESSES), q in {0.5,0.6,0.7,0.8,0.9,0.95,1.0}.
  Select: among pairs with calib failure-recall>=0.65, min calib P(fail|admit)
  (n_adm=0 -> P=1.0 anti-collapse); tie-break loosest (lowest TAU, then highest
  JMAX). If none>=0.65, max calib recall wins (same tie-breaks).
Validation: same splits/scorer as T199-T201 (frozen R173 calib/test/archive;
  pooled failure-reject recall x100 = pts; heldB yield = n_adm_success/20).
  Ablations: T199 ref (w>=TAU50), TAU-only at TAU_STAR (isolates veto gain),
  JMAX-only at JMAX_STAR (veto-alone signal).
Kill rule: DISCARD if pooled recall<=50 OR needs per-tool to pass (global-only
  by construction; second clause informational only).
Predicted 71.5 KEEP (+25 vs T199/T200 via recall recovery).
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
TAU_QS = [0.50, 0.60, 0.70, 0.80, 0.90]
JMAX_QS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
RECALL_FLOOR = 0.65


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

# ---- train-locked co-tune on CALIB ----
TAU_GRID = [qceil([w191(e) for e in cal_fails], q) for q in TAU_QS]
JMAX_GRID = [qceil([e["jerk"] for e in cal_succ], q) for q in JMAX_QS]


def calib_stats(tau, jmax):
    adm = [e for e in cal_eps if w191(e) >= tau and e["jerk"] <= jmax]
    n = len(adm)
    caught = sum(1 for e in cal_fails if not (w191(e) >= tau and e["jerk"] <= jmax))
    rec = caught / len(cal_fails)
    pf = sum(1 for e in adm if not e["success"]) / n if n else 1.0
    return {"recall": rec, "p_fail_adm": pf, "n_adm": n}


grid = {}
for ti, tau in enumerate(TAU_GRID):
    for ji, jmax in enumerate(JMAX_GRID):
        grid[(ti, ji)] = calib_stats(tau, jmax)

feasible = [(ti, ji) for (ti, ji), v in grid.items() if v["recall"] >= RECALL_FLOOR]
pool = feasible if feasible else list(grid.keys())
if feasible:
    best_pf = min(grid[k]["p_fail_adm"] for k in pool)
    cands = [k for k in pool if grid[k]["p_fail_adm"] == best_pf]
else:
    best_rec = max(grid[k]["recall"] for k in pool)
    cands = [k for k in pool if grid[k]["recall"] == best_rec]
cands.sort(key=lambda k: (TAU_GRID[k[0]], -JMAX_GRID[k[1]]))  # loosest
TI_STAR, JI_STAR = cands[0]
TAU_STAR, JMAX_STAR = TAU_GRID[TI_STAR], JMAX_GRID[JI_STAR]
edge_flag = (TI_STAR in (0, len(TAU_GRID) - 1)) or (JI_STAR in (0, len(JMAX_GRID) - 1))

a202 = lambda e: w191(e) >= TAU_STAR and e["jerk"] <= JMAX_STAR
a199 = lambda e: w191(e) >= EXP_TAU50
a_tau_only = lambda e: w191(e) >= TAU_STAR
a_jmax_only = lambda e: t173(e) and e["jerk"] <= JMAX_STAR

hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
assert len(hB) == 20
raw_keep = sum(1 for e in hB if e["success"]) / 20.0


def heldB_eval(fn):
    adm = [e for e in hB if fn(e)]
    s = sum(1 for e in adm if e["success"])
    n = len(adm)
    return {"n_adm": n, "yield_sel": round(s / 20.0, 4),
            "keep_pct": round(100 * n / 20.0, 2),
            "precision_adm": round(s / n, 4) if n else 1.0,
            "score_sel_pts": round(100 * s / n, 2) if n else 100.0,
            "cov_adm": round(st.mean(e["coverage_cont"] for e in adm), 4) if adm else 0.0}


def pooled_eval(fn):
    adm = [e for e in tst_eps if fn(e)]
    fails = [e for e in tst_eps if not e["success"]]
    caught = sum(1 for e in fails if not fn(e))
    n = len(adm)
    pf = sum(1 for e in adm if not e["success"]) / n if n else 0.0
    return {"n_adm": n, "p_fail_adm": round(pf, 4),
            "recall_fail_reject": round(caught / len(fails), 4),
            "score_recall_pts": round(100 * caught / len(fails), 2)}


e202 = heldB_eval(a202)
e199 = heldB_eval(a199)
e_tau = heldB_eval(a_tau_only)
e_jmx = heldB_eval(a_jmax_only)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
p202 = pooled_eval(a202)
p199 = pooled_eval(a199)
p_tau = pooled_eval(a_tau_only)
p_jmx = pooled_eval(a_jmax_only)

succ_rej199 = [e for e in tst_eps if e["success"] and not a199(e)]
rescued = [e for e in succ_rej199 if a202(e)]
rescue_rate = round(len(rescued) / len(succ_rej199), 4) if succ_rej199 else 0.0
new_mistakes = [e for e in tst_eps if not e["success"] and a202(e) and not a199(e)]
hB_rej199_succ = [e for e in hB if e["success"] and not a199(e)]
hB_rescued = [e for e in hB_rej199_succ if a202(e)]
vetoed_winners = [e for e in tst_eps if e["success"] and a_tau_only(e) and not a202(e)]

grid_out = {f"TAUq{q}|JMAXq{jq}": {"TAU": round(tau, 4), "JMAX": round(jmax, 6),
            "calib_recall": round(grid[(ti, ji)]["recall"], 4),
            "calib_p_fail_adm": round(grid[(ti, ji)]["p_fail_adm"], 4),
            "calib_n_adm": grid[(ti, ji)]["n_adm"]}
            for ti, (q, tau) in enumerate(zip(TAU_QS, TAU_GRID))
            for ji, (jq, jmax) in enumerate(zip(JMAX_QS, JMAX_GRID))}

out = {
    "variant": "T202 = global dual-cut: admit iff w191>=TAU AND jerk<=JMAX (2 params, global-only, no lam, no per-tool)",
    "frozen": {"TAU50_global_ref": EXP_TAU50, "theta_marg_bitident": True,
               "C99": round(C99, 4), "C99_degenerate": True,
               "calB_adm_n": len(calB_w), "cal_fails_n": len(cal_fails),
               "cal_succ_n": len(cal_succ),
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7),
               "fit_scope": "CALIB-only co-tune (TAU x JMAX grid, single held eval)"},
    "fit_calib_grid": grid_out,
    "locked": {"TAU_STAR": round(TAU_STAR, 4), "JMAX_STAR": round(JMAX_STAR, 6),
               "TAU_q": TAU_QS[TI_STAR], "JMAX_q": JMAX_QS[JI_STAR],
               "recall_floor": RECALL_FLOOR, "feasible_n": len(feasible),
               "star_at_grid_edge": edge_flag,
               "calib_star": {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in grid[(TI_STAR, JI_STAR)].items()}},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T202": e202, "ref_T199": e199, "TAU_only": e_tau,
              "JMAX_only": e_jmx, "cov_all": cov_all,
              "yield_vs_T199_pts": round(100 * (e202["yield_sel"] - e199["yield_sel"]), 2)},
    "pooled_held": {"T173": p_t173, "T202": p202, "ref_T199": p199,
                    "TAU_only": p_tau, "JMAX_only": p_jmx,
                    "veto_adds_vs_TAUonly_pts":
                        round(p202["score_recall_pts"] - p_tau["score_recall_pts"], 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - p202["p_fail_adm"]), 2),
                    "recall_vs_T199_pts":
                        round(p202["score_recall_pts"] - p199["score_recall_pts"], 2)},
    "rescue_vs_T199": {"held_succ_rejected_by_T199": len(succ_rej199),
                       "rescued_by_T202": len(rescued),
                       "rescue_rate": rescue_rate,
                       "new_mistakes_fails_adm_by_T202_not_T199": len(new_mistakes),
                       "heldB_succ_rejected_by_T199": len(hB_rej199_succ),
                       "heldB_rescued": len(hB_rescued),
                       "winners_vetoed_by_JMAX_tail": len(vetoed_winners),
                       "precision_no_drop_pooled": bool(p202["p_fail_adm"] <= p199["p_fail_adm"])},
    "archive_adm_frac_T202": round(sum(1 for e in arc_eps if a202(e)) / len(arc_eps), 4),
    "cov_ok": e202["cov_adm"] >= cov_all - 0.02,
    "director_predicted_71_5": p202["score_recall_pts"],
    "per_tool_needed": False,
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "kill_recall_le_50": p202["score_recall_pts"] <= 50.0,
    "recall_ge_70": p202["score_recall_pts"] >= 70.0,
    "heldB_precision_1": e202["precision_adm"] >= 1.0,
    "rescue_rate_gt_0": rescue_rate > 0,
    "precision_no_drop_vs_T199": out["rescue_vs_T199"]["precision_no_drop_pooled"],
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T202"] == 0.0,
    "global_only_no_pertool": True}
out["keep"] = bool(out["runA_keep_cited"] and not out["verdict_rule"]["kill_recall_le_50"]
                   and out["verdict_rule"]["recall_ge_70"]
                   and out["verdict_rule"]["heldB_precision_1"]
                   and out["verdict_rule"]["rescue_rate_gt_0"]
                   and out["verdict_rule"]["precision_no_drop_vs_T199"]
                   and out["verdict_rule"]["no_cov_regression"])
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r202_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
