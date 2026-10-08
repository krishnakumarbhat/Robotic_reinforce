"""Run 206: T199 = global-only ablation (director iter 24).

Director iter 24 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier kill: per-tool conformal line dead (T196 40.0 / T197 46.67 / T198 40.0,
all discard; per-tool w medians within 1.1pct -> no scale variance to fix;
shrinkage half-confirmed then killed).
Proposal T199 (pre-reg, single eval): global-only ablation, zero per-tool
tables, zero jerk gate:
  Admit199(e) := w191(e) >= TAU50_global
  w191 = min(T173/(max(slip,0.005)+0.005), C99), C99=100.0 degenerate,
  fric-free, T173-only (T173 embedded in w191 via w=0; NO extra THETA[t] check).
  TAU50_global = median(w191 | calib fitted-B admitted), train-locked,
  pooled 194+202 calibration (both drew from frozen R173 calib -> same pool),
  recomputed + asserted bit-identical to 68.0452 before eval.
Variation vs 203/204/205: REMOVES TAU_tool, TAU30_tool, THETA[t] entirely;
tests the overfit hypothesis (per-tool tables fit noise), not a 4th tuning.
Validation: same split/scorer as 204 (frozen R173 calib/test/archive;
pts = pooled failure-reject recall x100; keep% = heldB n_adm/20).
Win iff: recall_pts > 48 AND FP-tool variance drops vs T197 (per-tool line).
Predicted: 52-58pts (still <70, directional fix only).
Decision on fail: abandon w191-admission frontier entirely, move to new feature.
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
EXP_THETA_MARG = 0.014133
EXP_THETA = {0: 0.019941719703759794, 1: 0.007694750747550191,
             2: 0.01275873381323237}
EXP_TAU50_TOOL = {0: 76.30232298588356, 1: 76.26729063579836,
                  2: 75.47745665081642}
EXP_TAU30_TOOL = {0: 69.78764015236834, 1: 70.68624968849112,
                  2: 65.08405922571335}
EXP_NPT = {0: 29, 1: 40, 2: 39}
EXP_TAU50_S = {0: 72.9321, 1: 73.5266, 2: 72.9581}
EXP_TAU30_S = {0: 64.1278, 1: 65.7645, 2: 61.978}


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


def q193(xs, q):
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

cal_adm_raws = sorted(raw191(e) for e in cal_eps if t173(e))
C99 = q193(cal_adm_raws, 0.99)
assert abs(C99 - 100.0) < 1e-9, C99


def w191(e):
    r = raw191(e)
    return min(r, C99) if r > 0 else 0.0


# pooled 194+202 calibration pool == frozen R173 calib fitted-B admitted
calB = [e for e in cal_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
calB_w = sorted(w191(e) for e in calB if t173(e))
TAU50 = st.median(calB_w)
TAU30 = q193(calB_w, 0.30)
assert abs(TAU50 - EXP_TAU50) < 5e-5, TAU50
assert abs(TAU30 - EXP_TAU30) < 5e-5, TAU30

# per-tool tables recomputed ONLY as frozen refs for the variance ablation
THETA, TAU50_TOOL, TAU30_TOOL, NPT = {}, {}, {}, {}


def q194(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * len(s))))]


for t in (0, 1, 2):
    js = [e["jerk"] for e in cal_eps if e["tool_id"] == t and e["success"]]
    THETA[t] = q194(js, 0.9)
    assert abs(THETA[t] - EXP_THETA[t]) < 1e-9, (t, THETA[t])
    adm_all = sorted(w191(e) for e in cal_eps if e["tool_id"] == t and t173(e))
    NPT[t] = len(adm_all)
    assert NPT[t] == EXP_NPT[t], (t, NPT[t])
    TAU50_TOOL[t] = st.median(adm_all)
    TAU30_TOOL[t] = q193(adm_all, 0.30)
    assert abs(TAU50_TOOL[t] - EXP_TAU50_TOOL[t]) < 1e-9, (t, TAU50_TOOL[t])
    assert abs(TAU30_TOOL[t] - EXP_TAU30_TOOL[t]) < 1e-9, (t, TAU30_TOOL[t])
LAM = {t: 20.0 / (NPT[t] + 20.0) for t in (0, 1, 2)}
TAU50_S = {t: LAM[t] * TAU50 + (1 - LAM[t]) * TAU50_TOOL[t] for t in (0, 1, 2)}
TAU30_S = {t: LAM[t] * TAU30 + (1 - LAM[t]) * TAU30_TOOL[t] for t in (0, 1, 2)}
for t in (0, 1, 2):
    assert abs(TAU50_S[t] - EXP_TAU50_S[t]) < 5e-3, (t, TAU50_S[t])
    assert abs(TAU30_S[t] - EXP_TAU30_S[t]) < 5e-3, (t, TAU30_S[t])

a199 = lambda e: w191(e) >= TAU50  # global-only, zero per-tool, zero jerk gate
a197 = lambda e: (w191(e) >= TAU50_TOOL[e["tool_id"]]) or (  # run-204 ref
    w191(e) >= TAU30_TOOL[e["tool_id"]] and e["jerk"] <= THETA[e["tool_id"]])
a198 = lambda e: (w191(e) >= TAU50_S[e["tool_id"]]) or (  # run-205 ref
    w191(e) >= TAU30_S[e["tool_id"]] and e["jerk"] <= THETA[e["tool_id"]])

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


e199 = heldB_eval(a199)
e197 = heldB_eval(a197)
e198 = heldB_eval(a198)
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
p199 = pooled_eval(a199)
p197 = pooled_eval(a197)
p198 = pooled_eval(a198)


def per_tool_fpr(adm_fn):
    out = {}
    for t in (0, 1, 2):
        rows = [e for e in tst_eps if e["tool_id"] == t]
        a = [e for e in rows if adm_fn(e)]
        out[t] = round(sum(1 for e in a if not e["success"]) / len(a), 4) if a else 0.0
    return out


fpr199 = per_tool_fpr(a199)
fpr197 = per_tool_fpr(a197)
fpr198 = per_tool_fpr(a198)
var199 = round(st.pvariance(list(fpr199.values())), 6)
var197 = round(st.pvariance(list(fpr197.values())), 6)
var198 = round(st.pvariance(list(fpr198.values())), 6)

out = {
    "variant": "T199 = global-only ablation: admit iff w191>=TAU50_global (zero per-tool tables, zero jerk gate)",
    "frozen": {"TAU50_global": round(TAU50, 4), "TAU30_ref": round(TAU30, 4),
               "theta_marg_bitident": True,
               "C99": round(C99, 4), "C99_degenerate": True,
               "calB_adm_n": len(calB_w),
               "pooled_194_202_calib_is_R173": True,
               "pertool_refs_bitident": True, "N_calib_adm_tool": NPT,
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T199": e199, "ref_T197": e197, "ref_T198": e198, "cov_all": cov_all,
              "yield_vs_T197_pts": round(100 * (e199["yield_sel"] - e197["yield_sel"]), 2),
              "yield_vs_T198_pts": round(100 * (e199["yield_sel"] - e198["yield_sel"]), 2)},
    "pooled_held": {"T173": p_t173, "T199": p199, "ref_T197": p197, "ref_T198": p198,
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - p199["p_fail_adm"]), 2),
                    "recall_vs_T197_pts":
                        round(p199["score_recall_pts"] - p197["score_recall_pts"], 2),
                    "recall_vs_T198_pts":
                        round(p199["score_recall_pts"] - p198["score_recall_pts"], 2)},
    "per_tool_FPR_pooled": {"T199": fpr199, "ref_T197": fpr197, "ref_T198": fpr198},
    "FP_tool_variance": {"T199": var199, "ref_T197": var197, "ref_T198": var198,
                         "drops_vs_T197": bool(var199 < var197),
                         "drops_vs_T198": bool(var199 < var198)},
    "win_gt_48": bool(p199["score_recall_pts"] > 48.0),
    "archive_adm_frac_T199": round(sum(1 for e in arc_eps if a199(e)) / len(arc_eps), 4),
    "cov_ok": e199["cov_adm"] >= cov_all - 0.02,
    "precision_guard_heldB": e199["precision_adm"] >= 1.0,
    "director_predicted_52_58": p199["score_recall_pts"],
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "recall_gt_48": out["win_gt_48"],
    "fp_variance_drops_vs_T197": out["FP_tool_variance"]["drops_vs_T197"],
    "no_cov_regression": out["cov_ok"]}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r199_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
