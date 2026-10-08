"""Run 212: T205 = w191-only ablation, kill-test for jerk frontier (director iter 30).

Director iter 30 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: F-jerk exhausted (T202/T203/T204 all 46.67 = degenerate); close if tie.
Rule: Admit(e) iff w191(e) >= TAU*, NO jerk term, pooled-1 CALIB scope.
  w191(e) = min(1/(max(slip,0.005)+0.005), C99) if T173(e) else 0,
  T173(e) = jerk<=0.618 AND jerk<=THETA_MARG(0.014133) AND stall<=0.05.
Fit (train-locked, 5-fold stratified CV on CALIB only, NOT in-train co-tune):
  candidates TAU(q) = Qq of w191 over CALIB fails (q in 0.5..0.9 grid);
  per-fold: TAU fit on 4/5 train, recall+pf scored on held-out 1/5;
  TAU* = max mean-CV recall, tie-break loosest (smallest TAU), then min mean pf.
Single held eval after lock. Ablations/refs recomputed bit-identical:
  T199 (TAU50=68.0452), T202 (TAU=74.9317+JMAX=0.020572), T203 (bins E1/E2 +
  TAU_B), T204 (s=w-250*clip(jerk,0,0.015175)>=72.5843).
Reports: veto-rate, admit-overlap (agreement/Jaccard) vs T202-204 + T199,
  Pearson corr(w191,jerk) on CALIB + held (all eps, admitted-only).
Verdict: TIE at pooled recall 46.67 proves jerk adds zero -> DISCARD + CLOSE
  F-jerk, T206 = new orthogonal signal; WIN proves co-tune overfit.
Predicted 46.67pts, P(keep>=70)<5%. Frozen I7+I10. G7-clean, seg 15."""

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
T202_TAU, T202_JMAX = 74.9317, 0.020572
T203_E1, T203_E2 = 0.003652, 0.009075
T203_TAUB = [73.2251, 74.9317, 74.9317]
T204_LAM, T204_J95, T204_TAU = 250, 0.015175, 72.5843
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


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


def pearson(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


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
assert abs(st.median(calB_w) - EXP_TAU50) < 5e-5, st.median(calB_w)
assert len(calB_w) == 16, len(calB_w)
cal_fails = [e for e in cal_eps if not e["success"]]
assert len(cal_fails) == 17, len(cal_fails)

# ---- 5-fold stratified CV on CALIB (train-locked) ----
succ = [e for e in cal_eps if e["success"]]
fails = [e for e in cal_eps if not e["success"]]
folds = [[] for _ in range(5)]
for i, e in enumerate(succ):
    folds[i % 5].append(e)
for i, e in enumerate(fails):
    folds[(i * 2 + 1) % 5].append(e)
assert sum(len(f) for f in folds) == 120
assert all(sum(1 for e in f if not e["success"]) >= 3 for f in folds)

cand_tau_full = {q: qceil([w191(e) for e in cal_fails], q) for q in Q_GRID}
cv = {}
for q in Q_GRID:
    recs, pfs = [], []
    for v in range(5):
        tr = [e for j, f in enumerate(folds) for e in f if j != v]
        tr_fails = [e for e in tr if not e["success"]]
        tau = qceil([w191(e) for e in tr_fails], q)
        va = folds[v]
        va_fails = [e for e in va if not e["success"]]
        adm = [e for e in va if w191(e) >= tau]
        recs.append(sum(1 for e in va_fails if w191(e) < tau) / len(va_fails))
        pfs.append(sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0)
    cv[q] = {"TAU_full": round(cand_tau_full[q], 4),
             "mean_recall": round(st.mean(recs), 4),
             "mean_pf": round(st.mean(pfs), 4),
             "fold_recalls": [round(r, 4) for r in recs]}
best_rec = max(v["mean_recall"] for v in cv.values())
cands = [q for q in Q_GRID if cv[q]["mean_recall"] == best_rec]
cands.sort(key=lambda q: (cand_tau_full[q], cv[q]["mean_pf"]))
Q_STAR = cands[0]
TAU_STAR = cand_tau_full[Q_STAR]
edge_flag = Q_STAR in (Q_GRID[0], Q_GRID[-1])

# ---- locked rules ----
a205 = lambda e: w191(e) >= TAU_STAR  # noqa: E731
a199 = lambda e: w191(e) >= EXP_TAU50  # noqa: E731
a202 = lambda e: w191(e) >= T202_TAU and e["jerk"] <= T202_JMAX  # noqa: E731


def bin_of(e):
    j = e["jerk"]
    return 0 if j <= T203_E1 else (1 if j <= T203_E2 else 2)


a203 = lambda e: w191(e) >= T203_TAUB[bin_of(e)]  # noqa: E731
a204 = lambda e: (w191(e) - T204_LAM * min(max(e["jerk"], 0.0), T204_J95)) >= T204_TAU  # noqa: E731
rules = {"T199": a199, "T202": a202, "T203": a203, "T204": a204, "T205": a205}

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


e205, e199 = heldB_eval(a205), heldB_eval(a199)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = {name: pooled_eval(fn) for name, fn in rules.items()}


def overlap(fn_a, fn_b, eps):
    sa = {id(e) for e in eps if fn_a(e)}
    sb = {id(e) for e in eps if fn_b(e)}
    inter, union = len(sa & sb), len(sa | sb)
    agree = sum(1 for e in eps if fn_a(e) == fn_b(e)) / len(eps)
    return {"agreement": round(agree, 4),
            "jaccard": round(inter / union, 4) if union else 1.0,
            "n_a": len(sa), "n_b": len(sb), "n_both": inter}


ov_held = {f"T205_vs_{n}": overlap(a205, fn, tst_eps) for n, fn in rules.items() if n != "T205"}
ov_cal = {f"T205_vs_{n}": overlap(a205, fn, cal_eps) for n, fn in rules.items() if n != "T205"}

cw, cj = [w191(e) for e in cal_eps], [e["jerk"] for e in cal_eps]
hw, hj = [w191(e) for e in tst_eps], [e["jerk"] for e in tst_eps]
corr = {
    "calib_all": round(pearson(cw, cj), 4),
    "held_all": round(pearson(hw, hj), 4),
    "calib_admitted_T205": round(pearson(
        [w191(e) for e in cal_eps if a205(e)],
        [e["jerk"] for e in cal_eps if a205(e)]), 4),
    "held_admitted_T205": round(pearson(
        [w191(e) for e in tst_eps if a205(e)],
        [e["jerk"] for e in tst_eps if a205(e)]), 4),
}

succ_rej199 = [e for e in tst_eps if e["success"] and not a199(e)]
rescued = [e for e in succ_rej199 if a205(e)]
new_mist = [e for e in tst_eps if not e["success"] and a205(e) and not a199(e)]

out = {
    "variant": "T205 = w191-only ablation (admit iff w191>=TAU*, no jerk term, pooled-1 CALIB, 5-fold CV-tuned)",
    "frozen": {"TAU50_global_ref": EXP_TAU50, "theta_marg_bitident": True,
               "C99": round(C99, 4), "C99_degenerate": True,
               "calB_adm_n": len(calB_w), "cal_fails_n": len(cal_fails),
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7),
               "fit_scope": "5-fold stratified CV on CALIB only (pooled-1), single held eval"},
    "cv_tuning": {"Q_grid": Q_GRID,
                  "per_q": cv,
                  "Q_STAR": Q_STAR, "TAU_STAR": round(TAU_STAR, 4),
                  "q_at_grid_edge": bool(edge_flag)},
    "locked_refs": {"T202_TAU_JMAX": [T202_TAU, T202_JMAX],
                    "T203_E1_E2_TAUB": [T203_E1, T203_E2, T203_TAUB],
                    "T204_LAM_J95_TAU": [T204_LAM, T204_J95, T204_TAU]},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T205": e205, "ref_T199": e199, "cov_all": cov_all,
              "yield_vs_T199_pts": round(100 * (e205["yield_sel"] - e199["yield_sel"]), 2)},
    "pooled_held": {**pe,
                    "recall_vs_T199_pts": round(pe["T205"]["score_recall_pts"] - pe["T199"]["score_recall_pts"], 2),
                    "recall_vs_T202_pts": round(pe["T205"]["score_recall_pts"] - pe["T202"]["score_recall_pts"], 2),
                    "recall_vs_T203_pts": round(pe["T205"]["score_recall_pts"] - pe["T203"]["score_recall_pts"], 2),
                    "recall_vs_T204_pts": round(pe["T205"]["score_recall_pts"] - pe["T204"]["score_recall_pts"], 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["T205"]["p_fail_adm"]), 2)},
    "admit_overlap_held": ov_held,
    "admit_overlap_calib": ov_cal,
    "corr_w191_jerk": corr,
    "rescue_vs_T199": {"held_succ_rejected_by_T199": len(succ_rej199),
                       "rescued_by_T205": len(rescued),
                       "rescue_rate": round(len(rescued) / len(succ_rej199), 4) if succ_rej199 else 0.0,
                       "new_mistakes": len(new_mist)},
    "archive_adm_frac_T205": round(sum(1 for e in arc_eps if a205(e)) / len(arc_eps), 4),
    "cov_ok": e205["cov_adm"] >= cov_all - 0.02,
    "tie_vs_T202_T203_T204": bool(
        pe["T205"]["score_recall_pts"] == pe["T202"]["score_recall_pts"] == pe["T203"]["score_recall_pts"] == pe["T204"]["score_recall_pts"]),
    "director_predicted_pts_46.67": pe["T205"]["score_recall_pts"],
    "director_predicted_keep_ge_70": e205["keep_pct"],
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "HELD_keep_ge_70": e205["keep_pct"] >= 70.0,
    "pts_ge_70": pe["T205"]["score_recall_pts"] >= 70.0,
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T205"] == 0.0}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
out["frontier"] = ("CLOSE F-jerk (tie proves jerk adds zero); T206 = new orthogonal signal"
                   if out["tie_vs_T202_T203_T204"] else "WIN proves co-tune overfit; keep F-jerk open")
json.dump(out, open("results/aegis_v2/I2_r205_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
