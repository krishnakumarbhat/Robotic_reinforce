"""Run 208: T201 = global-only joint-tuned soft score (director iter 26).

Director iter 26 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: global linear w191 vs jerk tradeoff.
Rule: admit iff s = w191 - lam*jerk >= TAU_global (single lam, single TAU).
Variation vs T199/T200: same family as T200 (2-dof global linear score), but
  lam + TAU co-tuned in-train via 5-fold CV on CALIB only (nested: inner CV
  selects (lam,q); outer = single held-out pooled eval). No post-hoc refit on
  held; zero per-tool tables/buffer/shrinkage; zero rig edits; frozen R173 files.
  Unfrozen vs T200: theta_marg refit per CV-train fold (not a single frozen
  literal); I7=0.618 Tier-4 spec constant retained; TAU = Q_q over train-fold
  fails (q co-tuned, not fixed Q70).
Validation: nested CV (5-fold inner on calib) + held-out pooled; lam=0 ablation
  (same q*) to prove jerk adds; T199 hard-gate ref as second ablation.
Why: T198 per-tool overfit (40.0), T199/T200 underfit/frozen (46.67); 2-dof
  global joint fit fixes both.
Predicted 70-73 KEEP. Discard iff <60 or lam unstable to 0/inf.
G7-clean, seg 15."""

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
LAM_GRID = [0, 100, 250, 500, 1000, 2000, 4000, 8000]
Q_GRID = [0.50, 0.60, 0.70, 0.80, 0.90]
N_FOLDS = 5
CV_SEED = 42


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

# Bit-identical recompute on FULL calib (asserts before eval, T-series precedent)
sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * 0.9) - 1)
TH_MARG_FULL = sj[k]
assert abs(TH_MARG_FULL - 0.014133) < 1e-6, TH_MARG_FULL
cal_adm_raws = sorted(
    1.0 / (max(e["slip_m"], FLOOR) + EPS)
    for e in cal_eps
    if e["jerk"] <= I7 and e["jerk"] <= TH_MARG_FULL and e["stall_frac"] <= STALL_CAP)
C99_FULL = qceil(cal_adm_raws, 0.99)
assert abs(C99_FULL - 100.0) < 1e-9, C99_FULL


def make_scorer(theta_marg, c99):
    def t173(e):
        return e["jerk"] <= I7 and e["jerk"] <= theta_marg and e["stall_frac"] <= STALL_CAP

    def w191(e):
        if not t173(e):
            return 0.0
        return min(1.0 / (max(e["slip_m"], FLOOR) + EPS), c99)

    def s_of(e, lam):
        return w191(e) - lam * e["jerk"]

    return t173, w191, s_of


_, w191_full, s_full = make_scorer(TH_MARG_FULL, C99_FULL)
calB_full = [e for e in cal_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
calB_w = sorted(w191_full(e) for e in calB_full if e["jerk"] <= I7 and e["jerk"] <= TH_MARG_FULL and e["stall_frac"] <= STALL_CAP)
assert abs(st.median(calB_w) - EXP_TAU50) < 5e-5, st.median(calB_w)
assert len(calB_w) == 16, len(calB_w)

# ---- 5-fold CV on CALIB (stratified by success, seeded) ----
idx_s = [i for i, e in enumerate(cal_eps) if e["success"]]
idx_f = [i for i, e in enumerate(cal_eps) if not e["success"]]
import random
rng = random.Random(CV_SEED)
rng.shuffle(idx_s)
rng.shuffle(idx_f)
folds = [[] for _ in range(N_FOLDS)]
for j, i in enumerate(idx_s):
    folds[j % N_FOLDS].append(i)
for j, i in enumerate(idx_f):
    folds[j % N_FOLDS].append(i)

cal_fails_full = [e for e in cal_eps if not e["success"]]
assert len(cal_fails_full) == 17, len(cal_fails_full)

cv_table = {}  # (lam,q) -> list of val recall
cv_prec = {}  # (lam,q) -> list of val P(fail|admit)
fold_best_lam_q70 = []  # stability probe: best lam per fold at q=0.70
for lam in LAM_GRID:
    for q in Q_GRID:
        recs, precs = [], []
        for v in range(N_FOLDS):
            tr = [cal_eps[i] for f, fl in enumerate(folds) if f != v for i in fl]
            va = [cal_eps[i] for i in folds[v]]
            tr_s = sorted(e["jerk"] for e in tr if e["success"])
            kk = min(len(tr_s) - 1, math.ceil((len(tr_s) + 1) * 0.9) - 1)
            th_tr = tr_s[kk]
            raw_tr = sorted(1.0 / (max(e["slip_m"], FLOOR) + EPS)
                            for e in tr if e["jerk"] <= I7 and e["jerk"] <= th_tr and e["stall_frac"] <= STALL_CAP)
            c_tr = qceil(raw_tr, 0.99)
            _, w_tr, s_tr = make_scorer(th_tr, c_tr)
            tr_fails = [e for e in tr if not e["success"]]
            tau = qceil([s_tr(e, lam) for e in tr_fails], q)
            adm = [e for e in va if s_tr(e, lam) >= tau]
            vfails = [e for e in va if not e["success"]]
            caught = sum(1 for e in vfails if s_tr(e, lam) < tau)
            recs.append(caught / len(vfails) if vfails else 1.0)
            precs.append(sum(1 for e in adm if not e["success"]) / len(adm) if adm else 0.0)
        cv_table[(lam, q)] = recs
        cv_prec[(lam, q)] = precs

mean_rec = {k: st.mean(v) for k, v in cv_table.items()}
mean_pf = {k: st.mean(v) for k, v in cv_prec.items()}
best_rec = max(mean_rec.values())
cands = [k for k, v in mean_rec.items() if v == best_rec]
# tie-break: lowest mean P(fail|admit), then smallest lam, then q closest to 0.70
cands.sort(key=lambda k: (mean_pf[k], k[0], abs(k[1] - 0.70)))
LAM_STAR, Q_STAR = cands[0]
TAU_STAR = qceil([s_full(e, LAM_STAR) for e in cal_fails_full], Q_STAR)

# stability probe: per-fold best lam at q=0.70 (fit on other 4 folds, same metric)
for v in range(N_FOLDS):
    tr = [cal_eps[i] for f, fl in enumerate(folds) if f != v for i in fl]
    va = [cal_eps[i] for i in folds[v]]
    tr_s = sorted(e["jerk"] for e in tr if e["success"])
    kk = min(len(tr_s) - 1, math.ceil((len(tr_s) + 1) * 0.9) - 1)
    th_tr = tr_s[kk]
    raw_tr = sorted(1.0 / (max(e["slip_m"], FLOOR) + EPS)
                    for e in tr if e["jerk"] <= I7 and e["jerk"] <= th_tr and e["stall_frac"] <= STALL_CAP)
    c_tr = qceil(raw_tr, 0.99)
    _, _, s_tr = make_scorer(th_tr, c_tr)
    tr_fails = [e for e in tr if not e["success"]]
    per_lam = {}
    for lam in LAM_GRID:
        tau = qceil([s_tr(e, lam) for e in tr_fails], 0.70)
        vfails = [e for e in va if not e["success"]]
        caught = sum(1 for e in vfails if s_tr(e, lam) < tau)
        per_lam[lam] = caught / len(vfails) if vfails else 1.0
    bl = max(per_lam.values())
    fold_best_lam_q70.append(min(lam for lam, r in per_lam.items() if r == bl))
lam_unstable = (all(l == 0 for l in fold_best_lam_q70)
                or all(l == LAM_GRID[-1] for l in fold_best_lam_q70))

# ---- single held-out eval ----
t173_full, _, _ = make_scorer(TH_MARG_FULL, C99_FULL)
a199 = lambda e: w191_full(e) >= EXP_TAU50
adm201 = lambda e: s_full(e, LAM_STAR) >= TAU_STAR
TAU_LAM0 = qceil([s_full(e, 0) for e in cal_fails_full], Q_STAR)
a201_lam0 = lambda e: s_full(e, 0) >= TAU_LAM0

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


e201 = heldB_eval(adm201)
e199 = heldB_eval(a199)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173_full)
p201 = pooled_eval(adm201)
p199 = pooled_eval(a199)
p_lam0 = pooled_eval(a201_lam0)

succ_rej199 = [e for e in tst_eps if e["success"] and not a199(e)]
rescued = [e for e in succ_rej199 if adm201(e)]
rescue_rate = round(len(rescued) / len(succ_rej199), 4) if succ_rej199 else 0.0
new_mistakes = [e for e in tst_eps if not e["success"] and adm201(e) and not a199(e)]
hB_rej199_succ = [e for e in hB if e["success"] and not a199(e)]
hB_rescued = [e for e in hB_rej199_succ if adm201(e)]

cv_summary = {f"{lam}@{q}": {"mean_recall": round(mean_rec[(lam, q)], 4),
                             "mean_p_fail_adm": round(mean_pf[(lam, q)], 4)}
              for lam in LAM_GRID for q in Q_GRID}

out = {
    "variant": "T201 = global-only joint-tuned soft score: s=w191-lam*jerk, admit iff s>=TAU_global (single lam+TAU, 5-fold CV in-train, zero per-tool tables)",
    "frozen": {"theta_marg_full_bitident": True, "C99_full": round(C99_FULL, 4),
               "C99_degenerate": True, "TAU50_global_ref": EXP_TAU50,
               "cal_fails_n": len(cal_fails_full), "calB_adm_n": len(calB_w),
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7),
               "unfrozen_vs_T200": "theta_marg refit per CV-train fold; TAU=Q_q co-tuned (not fixed Q70); I7=0.618 Tier-4 spec constant retained",
               "fit_scope": "CALIB-only 5-fold stratified CV (seed 42); single held eval (nested)"},
    "cv": {"n_folds": N_FOLDS, "seed": CV_SEED, "lam_grid": LAM_GRID, "q_grid": Q_GRID,
           "grid": cv_summary,
           "best_mean_recall": round(best_rec, 4),
           "fold_best_lam_at_q70": fold_best_lam_q70,
           "lam_unstable_to_0_or_inf": lam_unstable},
    "locked": {"LAM_STAR": LAM_STAR, "Q_STAR": Q_STAR, "TAU_STAR": round(TAU_STAR, 4),
               "TAU_LAM0_same_q": round(TAU_LAM0, 4)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T201": e201, "ref_T199": e199, "cov_all": cov_all,
              "yield_vs_T199_pts": round(100 * (e201["yield_sel"] - e199["yield_sel"]), 2)},
    "pooled_held": {"T173": p_t173, "T201": p201, "ref_T199": p199,
                    "ref_lam0_same_q": p_lam0,
                    "jerk_adds_vs_lam0_pts": round(p201["score_recall_pts"] - p_lam0["score_recall_pts"], 2),
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - p201["p_fail_adm"]), 2),
                    "recall_vs_T199_pts":
                        round(p201["score_recall_pts"] - p199["score_recall_pts"], 2)},
    "rescue_vs_T199": {"held_succ_rejected_by_T199": len(succ_rej199),
                       "rescued_by_T201": len(rescued),
                       "rescue_rate": rescue_rate,
                       "new_mistakes_fails_adm_by_T201_not_T199": len(new_mistakes),
                       "heldB_succ_rejected_by_T199": len(hB_rej199_succ),
                       "heldB_rescued": len(hB_rescued),
                       "precision_no_drop_pooled": bool(p201["p_fail_adm"] <= p199["p_fail_adm"])},
    "archive_adm_frac_T201": round(sum(1 for e in arc_eps if adm201(e)) / len(arc_eps), 4),
    "cov_ok": e201["cov_adm"] >= cov_all - 0.02,
    "director_predicted_70_73": p201["score_recall_pts"],
}
out["verdict_rule"] = {
    "recall_ge_60": p201["score_recall_pts"] >= 60.0,
    "lam_stable": not lam_unstable,
    "recall_ge_70": p201["score_recall_pts"] >= 70.0,
    "heldB_precision_1": e201["precision_adm"] >= 1.0,
    "rescue_rate_gt_0": rescue_rate > 0,
    "precision_no_drop_vs_T199": out["rescue_vs_T199"]["precision_no_drop_pooled"],
    "jerk_adds_gt_0": out["pooled_held"]["jerk_adds_vs_lam0_pts"] > 0,
    "no_cov_regression": out["cov_ok"],
    "archive_zero": out["archive_adm_frac_T201"] == 0.0}
out["director_discard_fires"] = bool(p201["score_recall_pts"] < 60.0 or lam_unstable)
out["keep"] = bool(out["verdict_rule"]["recall_ge_70"] and out["verdict_rule"]["lam_stable"]
                   and out["verdict_rule"]["heldB_precision_1"] and out["verdict_rule"]["rescue_rate_gt_0"]
                   and out["verdict_rule"]["precision_no_drop_vs_T199"]
                   and out["verdict_rule"]["jerk_adds_gt_0"] and out["verdict_rule"]["no_cov_regression"]
                   and out["runA_keep_cited"])
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r201_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
