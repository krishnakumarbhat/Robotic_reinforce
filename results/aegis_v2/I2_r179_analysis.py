"""Run 179 offline analysis: T179 low-fric-tolerant shrinker, POST-HOC ONLY.

Director iter 13 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: soft-shrink calibration (continuous down-weight > hard admit).
Variation vs T178: halve fric penalty again (1-0.5*fric)->(1-0.25*fric);
  w179(e) = 1[T173(e)] * P50/(slip_m+P50) * (1-0.25*fric),
  P50=0.008106 frozen R175 train-median (6dp assert, never refit).
Why: T176 hard AND discard brittle; T177 full (1-fric) over-penalizes -> P5;
  T178 half-penalty still -0.85pts vs T173 -> fric term still dominant.
Pre-reg verdict: KEEP iff runA_keep (cited) AND score_w179 >= 70 AND
  no coverage regression (cov_w >= cov_all - 0.02). Lifts reported UNGATED.
Ablation (isolate fric vs slip): alpha in {0, 0.25, 0.5} x
  P50 in {0.008 (rounded), median 0.008106} = 6 cells, held-out pooled only.
  alpha=0 -> slip-kernel-only (no fric term); P50 rounded vs exact -> rounding
  sensitivity (expect ~0: proves P50 lock inert).
Frozen inputs (zero rig edits, no rig run): R173 evidence files.
G7: coverage/success from physics logs only; weights scale confidence mass,
  never per-episode coverage/success (aggregates only, as R174-R178)."""

import json
import math
import statistics

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
ALPHA = 0.1
I7_THETA = 0.618
STALL_CAP = 0.05
P50_MEDIAN = 0.008106  # frozen R175 train-median (6dp)
P50_ROUNDED = 0.008  # ablation rounding probe
ALPHA_179 = 0.25  # director pre-reg fric coefficient


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def w_alpha(e, theta, alpha, p50):
    if not t173(e, theta):
        return 0.0
    return (p50 / (e["slip_m"] + p50)) * (1.0 - alpha * e["friction"])


def w177(e, theta):
    return w_alpha(e, theta, 1.0, P50_MEDIAN)


def w178(e, theta):
    return w_alpha(e, theta, 0.5, P50_MEDIAN)


def w179(e, theta):
    return w_alpha(e, theta, ALPHA_179, P50_MEDIAN)


def hard_rule(eps, admit_fn):
    suc = [e for e in eps if e["success"]]
    adm = [e for e in eps if admit_fn(e)]
    adm_s = sum(1 for e in adm if e["success"])
    rec = adm_s / len(suc) if suc else 1.0
    prec = adm_s / len(adm) if adm else 1.0
    p_fail = (len(adm) - adm_s) / len(adm) if adm else 0.0
    cov_adm = sum(e["coverage_cont"] for e in adm) / len(adm) if adm else 0.0
    cov_all = sum(e["coverage_cont"] for e in eps) / len(eps)
    return {"n": len(eps), "n_adm": len(adm),
            "recall": round(rec, 4), "score": round(100 * rec, 2),
            "precision": round(prec, 4),
            "p_fail_given_adm": round(p_fail, 4),
            "cov_adm": round(cov_adm, 4), "cov_all": round(cov_all, 4)}


def soft_eval(eps, theta, wfn):
    ws = [wfn(e, theta) for e in eps]
    assert all(0.0 <= w < 1.0 + 1e-12 for w in ws), "weight out of [0,1)"
    assert all((w == 0.0) == (not t173(e, theta))
               for w, e in zip(ws, eps)), "T173 support mismatch"
    sw = sum(ws)
    sw2 = sum(w * w for w in ws)
    fail_w = sum(w for w, e in zip(ws, eps) if not e["success"])
    succ_w = sw - fail_w
    n_succ = sum(1 for e in eps if e["success"])
    p_w = fail_w / sw if sw else 0.0
    prec_w = succ_w / sw if sw else 1.0
    rec_w = succ_w / n_succ if n_succ else 1.0
    cov_w = sum(w * e["coverage_cont"] for w, e in zip(ws, eps)) / sw if sw else 0.0
    cov_all = sum(e["coverage_cont"] for e in eps) / len(eps)
    ess = sw * sw / sw2 if sw2 else 0.0
    return {"n": len(eps), "sum_w": round(sw, 4),
            "mean_w": round(sw / len(eps), 4),
            "ess": round(ess, 2),
            "p_fail_weighted": round(p_w, 4),
            "precision_w": round(prec_w, 4),
            "score_w": round(100 * prec_w, 2),
            "recall_w": round(rec_w, 4),
            "cov_w": round(cov_w, 4), "cov_all": round(cov_all, 4)}


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)
assert cal_hdr["pose_noise_cfg"] == "0.01,2" and cal_hdr["gate_mode"] == "post-hoc"
assert tst_hdr["pose_noise_cfg"] == "0.01,2" and arc_hdr["pose_noise_cfg"] == "2.0,2"
assert cal_hdr["path_mode"] == "fitted" and cal_hdr["compare"] == "trochoid"

# Train-fold pre-reg locks (LOG-ONLY identity checks; frozen, never refit)
sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * (1 - ALPHA)) - 1)
theta_marg = sj[k]
assert abs(theta_marg - 0.014133) < 1e-6, f"rig drift: theta={theta_marg}"
succ_slip = sorted(e["slip_m"] for e in cal_eps if e["success"])
p50_train = statistics.median(succ_slip)
assert round(p50_train, 6) == P50_MEDIAN, "slip P50 drift"
assert abs(P50_ROUNDED - P50_MEDIAN) < 2e-4, "rounding probe too far"

# Spot check: manual recompute of w179 for first T173-admitted calib episode
spot = next(e for e in cal_eps if t173(e, theta_marg))
manual = (P50_MEDIAN / (spot["slip_m"] + P50_MEDIAN)) * (1.0 - ALPHA_179 * spot["friction"])
assert abs(w179(spot, theta_marg) - manual) < 1e-15, "weight formula mismatch"
# Identity: w179/w178 ratio must equal (1-0.25f)/(1-0.5f) exactly
for e in cal_eps[:5]:
    if t173(e, theta_marg) and e["friction"] < 2.0:
        r = w179(e, theta_marg) / w178(e, theta_marg)
        assert abs(r - (1 - 0.25 * e["friction"]) / (1 - 0.5 * e["friction"])) < 1e-12

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "n_success_calib": len(sj),
    "P50_slip_frozen_trainonly": P50_MEDIAN,
    "P50_train_recomputed": round(p50_train, 6),
    "alpha_179": ALPHA_179,
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0,
               cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "spot_check": {"seed": spot["seed"], "suite": spot["suite"],
                   "slip": spot["slip_m"], "fric": spot["friction"],
                   "w179": round(w179(spot, theta_marg), 6),
                   "w178": round(w178(spot, theta_marg), 6),
                   "w177": round(w177(spot, theta_marg), 6)},
    "trainfold_T173": hard_rule(cal_eps, lambda e: t173(e, theta_marg)),
    "trainfold_T179": soft_eval(cal_eps, theta_marg, w179),
    "held_T173": hard_rule(tst_eps, lambda e: t173(e, theta_marg)),
    "held_T177": soft_eval(tst_eps, theta_marg, w177),
    "held_T178": soft_eval(tst_eps, theta_marg, w178),
    "held_T179": soft_eval(tst_eps, theta_marg, w179),
}
h3, h7, h8, h9 = out["held_T173"], out["held_T177"], out["held_T178"], out["held_T179"]
t9 = out["trainfold_T179"]
out["lift_T179_vs_T173_pts"] = round(100 * (h3["p_fail_given_adm"] - h9["p_fail_weighted"]), 2)
out["lift_T179_vs_T177_pts"] = round(100 * (h7["p_fail_weighted"] - h9["p_fail_weighted"]), 2)
out["lift_T179_vs_T178_pts"] = round(100 * (h8["p_fail_weighted"] - h9["p_fail_weighted"]), 2)
out["train_lift_T179_vs_T173_pts"] = round(
    100 * (out["trainfold_T173"]["p_fail_given_adm"] - t9["p_fail_weighted"]), 2)
out["train_lift_T179_vs_T178_pts"] = round(
    100 * (soft_eval(cal_eps, theta_marg, w178)["p_fail_weighted"] - t9["p_fail_weighted"]), 2)
out["cov_ok"] = h9["cov_w"] >= h9["cov_all"] - 0.02

# Ablation grid: alpha x P50 on held-out pooled (LOG-ONLY, single eval each)
abl = {}
for a in (0, 0.25, 0.5):
    for pname, p50 in (("P50rnd0.008", P50_ROUNDED), ("P50median", P50_MEDIAN)):
        cell = soft_eval(tst_eps, theta_marg, lambda e, th, aa=a, pp=p50: w_alpha(e, th, aa, pp))
        abl[f"alpha{a}_x_{pname}"] = {
            "p_fail_weighted": cell["p_fail_weighted"],
            "precision_w": cell["precision_w"], "score_w": cell["score_w"],
            "mean_w": cell["mean_w"], "ess": cell["ess"]}
out["ablation_alpha_x_P50_held"] = abl
f0m = abl["alpha0_x_P50median"]
out["ablation_sliponly_lift_vs_T173_pts"] = round(100 * (h3["p_fail_given_adm"] - f0m["p_fail_weighted"]), 2)
out["ablation_P50rounding_delta_pts"] = round(
    100 * abs(abl["alpha0.25_x_P50median"]["p_fail_weighted"] - abl["alpha0.25_x_P50rnd0.008"]["p_fail_weighted"]), 3)
out["ablation_alpha_slope_per0.25_pts"] = round(
    100 * (abl["alpha0.5_x_P50median"]["p_fail_weighted"] - abl["alpha0_x_P50median"]["p_fail_weighted"]) / 2, 3)

# Archive valid-mass (LOG-ONLY): soft weights never fully abstain
arc_ws = [w179(e, theta_marg) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["archive_T173_valid_rate"] = round(
    sum(1 for e in arc_eps if t173(e, theta_marg)) / len(arc_eps), 4)
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "score_w_ge_70": h9["score_w"] >= 70.0,
    "no_cov_regression": out["cov_ok"],
}
out["keep"] = bool(all(out["verdict_rule"].values()))
json.dump(out, open("results/aegis_v2/I2_r179_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
