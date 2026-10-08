"""Run 203: T196 = per-tool conformal admission, relaxed hybrid of 194+202 (director iter 21).

Director iter 21 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: admission gating (T193-T195 exhausted global thresholds; best 33.33pts, keep>=70 unmet).
Variation (pre-reg, single eval): per-tool conformal admission, T173 kept as pre-filter only:
  Admit196(e) := w191(e) >= TAU_tool[t]
              OR (w191(e) >= TAU30_tool[t] AND jerk(e) <= THETA[t]), t = tool_id(e)
  w191 = min(T173/(max(slip,0.005)+0.005), C99), C99=100.0 degenerate,
  fric-free, T173-only, damping ON, frozen I7+I10.
  Train-locked (recomputed + asserted bit-identical before eval):
  TAU_tool[t]  = Q10(w191 | calib, tool=t, T173-admitted, SUCCESS) = 90% recall
                 of calib successes per tool (conformal, ceil-quantile, r193 method);
  TAU30_tool[t]= Q30(w191 | calib, tool=t, T173-admitted, ALL) = per-tool buffer
                 floor (same quantile method as global TAU30);
  THETA[t] = Q90(jerk | calib SUCCESS, tool=t) (r194 int-quantile, recomputed +
             asserted bit-identical to frozen {t0:0.01994172,t1:0.00769475,t2:0.01275873});
  theta_marg = 0.014133 (T173 prefilter, exact float), I7 ceiling 0.618, stall cap 0.05.
Why: T194 POST-HOC ONLY too late/strict (+0.37pts); T195 global TAU50=68.0452 too
  harsh (vetoes successes only, tool-1 kill); per-tool TAUs fix cross-tool scale
  variance while keeping the T173 prefilter.
Validation: same suite as Run 202 (frozen R173 calib/test/archive, frozen seeds);
  report keep (heldB yield) + pts (pooled failure-recall score) + FPR (p_fail_adm)
  per tool vs T194 baseline.
Predicted: 68-73pts, P(>=70)~55%; abort/discard if score<50.
Kill iff: score_recall < 50 (abort) OR any tool T196 p_fail_adm_tool > T194
  baseline same tool (same kill as T195).
Frozen I7+I10, zero rig edits, no rig run, no refit. G7-clean, seg 15."""

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
EXP_THETA = {0: 0.019941719703759794, 1: 0.007694750747550191,
             2: 0.01275873381323237}
EXP_THETA_MARG = 0.014133


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, th=None):
    th = TH_MARG if th is None else th
    return e["jerk"] <= I7 and e["jerk"] <= th and e["stall_frac"] <= STALL_CAP


def raw191(e):
    return 1.0 / (max(e["slip_m"], FLOOR) + EPS) if t173(e) else 0.0


def q193(xs, q):  # r193 ceil-quantile (conformal)
    s = sorted(xs)
    i = min(len(s) - 1, math.ceil(q * len(s)) - 1)
    return s[i]


def q194(xs, q):  # r194 int-quantile
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * len(s))))]


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)
assert cal_hdr["pose_noise_cfg"] == "0.01,2" and cal_hdr["gate_mode"] == "post-hoc"
assert tst_hdr["pose_noise_cfg"] == "0.01,2" and arc_hdr["pose_noise_cfg"] == "2.0,2"

sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * 0.9) - 1)
TH_MARG = sj[k]
assert abs(TH_MARG - EXP_THETA_MARG) < 1e-6, TH_MARG

cal_adm_raws = sorted(raw191(e) for e in cal_eps if t173(e))
C99 = q193(cal_adm_raws, 0.99)
assert abs(C99 - 100.0) < 1e-9, C99


def w191(e):
    r = raw191(e)
    return min(r, C99) if r > 0 else 0.0


THETA, NB = {}, {}
for t in (0, 1, 2):
    js = [e["jerk"] for e in cal_eps if e["tool_id"] == t and e["success"]]
    NB[t] = len(js)
    THETA[t] = q194(js, 0.9)
assert all(n >= 20 for n in NB.values()), NB
for t in (0, 1, 2):
    assert abs(THETA[t] - EXP_THETA[t]) < 1e-9, (t, THETA[t])

# global refs (log-only, bit-identity vs frozen T192/T193)
calB = [e for e in cal_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
calB_w = sorted(w191(e) for e in calB if t173(e))
TAU50 = st.median(calB_w)
TAU30 = q193(calB_w, 0.30)
assert abs(TAU50 - EXP_TAU50) < 5e-5, TAU50
assert abs(TAU30 - EXP_TAU30) < 5e-5, TAU30

# --- T196 per-tool conformal thresholds (decider, train-locked) ---
TAU_TOOL, TAU30_TOOL, NPT = {}, {}, {}
for t in (0, 1, 2):
    adm_all = sorted(w191(e) for e in cal_eps if e["tool_id"] == t and t173(e))
    adm_succ = sorted(w191(e) for e in cal_eps
                      if e["tool_id"] == t and t173(e) and e["success"])
    NPT[t] = {"n_adm": len(adm_all), "n_adm_succ": len(adm_succ)}
    assert len(adm_all) >= 10 and len(adm_succ) >= 10, (t, NPT[t])
    TAU_TOOL[t] = q193(adm_succ, 0.10)    # 90% recall of calib successes, per tool
    TAU30_TOOL[t] = q193(adm_all, 0.30)   # per-tool buffer floor

# --- admission rules ---
a194 = lambda e: t173(e) and e["jerk"] <= THETA[e["tool_id"]]
a196 = lambda e: (w191(e) >= TAU_TOOL[e["tool_id"]]) or (
    w191(e) >= TAU30_TOOL[e["tool_id"]] and e["jerk"] <= THETA[e["tool_id"]])

hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
assert len(hB) == 20
raw_keep = sum(1 for e in hB if e["success"]) / 20.0


def heldB_eval(adm_fn):
    adm = [e for e in hB if adm_fn(e)]
    s = sum(1 for e in adm if e["success"])
    n_adm = len(adm)
    return {"n_adm": n_adm,
            "yield_sel": round(s / 20.0, 4),
            "keep_pct": round(100 * n_adm / 20.0, 2),
            "precision_adm": round(s / n_adm, 4) if n_adm else 1.0,
            "score_sel_pts": round(100 * s / n_adm, 2) if n_adm else 100.0,
            "cov_adm": round(st.mean(e["coverage_cont"] for e in adm), 4) if adm else 0.0}


e196 = heldB_eval(a196)
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
p_194 = pooled_eval(a194)
p_196 = pooled_eval(a196)

per_tool = {}
kill_tool = []
for t in (0, 1, 2):
    rows = [e for e in tst_eps if e["tool_id"] == t]
    a4 = [e for e in rows if a194(e)]
    a6 = [e for e in rows if a196(e)]

    def pf(rs):
        return sum(1 for e in rs if not e["success"]) / len(rs) if rs else 0.0

    pf4, pf6 = pf(a4), pf(a6)
    fires = pf6 > pf4 + 1e-12
    kill_tool.append(fires)
    succ = [e for e in rows if e["success"]]
    rec6 = sum(1 for e in succ if a196(e)) / len(succ) if succ else 1.0
    per_tool[t] = {"n": len(rows),
                   "TAU_tool": TAU_TOOL[t], "TAU30_tool": TAU30_TOOL[t],
                   "THETA_tool": THETA[t], "n_calib": NPT[t],
                   "T194": {"n_adm": len(a4), "p_fail_adm_FPR": round(pf4, 4)},
                   "T196": {"n_adm": len(a6), "p_fail_adm_FPR": round(pf6, 4),
                            "success_recall": round(rec6, 4)},
                   "kill_tool_violation_gt_194": bool(fires)}

out = {
    "variant": "T196 = per-tool conformal admission: admit iff w191>=TAU_tool[t] OR (w191>=TAU30_tool[t] AND jerk<=THETA[t])",
    "frozen": {"theta_marg_bitident": True, "C99": round(C99, 4),
               "C99_degenerate": True,
               "TAU50_global_ref": round(TAU50, 4),
               "TAU30_global_ref": round(TAU30, 4),
               "TAU_tool_per_tool": {str(kk): vv for kk, vv in TAU_TOOL.items()},
               "TAU30_tool_per_tool": {str(kk): vv for kk, vv in TAU30_TOOL.items()},
               "THETA_tool_bitident": {str(kk): vv for kk, vv in THETA.items()},
               "N_calib_succ_tool": NB, "calB_adm_n": len(calB_w),
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T196": e196, "cov_all": cov_all},
    "pooled_held": {"T173": p_t173, "T194": p_194, "T196": p_196,
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - p_196["p_fail_adm"]), 2)},
    "per_tool_held_pooled": per_tool,
    "kill_any_tool_violation_gt_194": bool(any(kill_tool)),
    "abort_score_lt_50": bool(p_196["score_recall_pts"] < 50.0),
    "archive_adm_frac_T196": round(sum(1 for e in arc_eps if a196(e)) / len(arc_eps), 4),
    "cov_ok": e196["cov_adm"] >= cov_all - 0.02,
    "precision_guard_heldB": e196["precision_adm"] >= 1.0,
    "director_predicted_score_68_73": p_196["score_recall_pts"],
    "director_predicted_keep_ge_70": e196["keep_pct"],
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "score_recall_ge_70": p_196["score_recall_pts"] >= 70.0,
    "precision_guard": out["precision_guard_heldB"],
    "no_cov_regression": out["cov_ok"],
    "no_kill": not (out["abort_score_lt_50"] or out["kill_any_tool_violation_gt_194"])}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r196_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
