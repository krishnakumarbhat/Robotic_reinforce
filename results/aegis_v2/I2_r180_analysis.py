"""Run 180 offline analysis: T180 zero-fric control, POST-HOC ONLY.

Director iter 14 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: post-hoc soft shrinker (close fric-penalty line).
Variation vs T177/T178/T179: drop fric factor entirely (k=1.0/0.5/0.25 -> k=0);
  w180(e) = 1[T173(e)] * P50/(slip_m+P50),
  P50=0.008 frozen (director lock, no refit; cf R175 median 0.008106).
Why: decisive kill-test; if k=0 still ~-0.5pt vs T173, fric is not the
  blocker (slip kernel alone carries no signal either) -> pivot frontier next.
Pre-reg verdict: KEEP iff runA_keep (cited) AND score_w180 >= 70 AND
  no coverage regression (cov_w >= cov_all - 0.02). Lifts reported UNGATED.
Frozen inputs (zero rig edits, no rig run): R173 evidence files.
G7: coverage/success from physics logs only; weights scale confidence mass,
  never per-episode coverage/success (aggregates only, as R174-R179)."""

import json
import math
import statistics

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
ALPHA = 0.1
I7_THETA = 0.618
STALL_CAP = 0.05
P50_FROZEN = 0.008  # director lock (cf R175 median 0.008106, delta <2e-4 inert per R179)
P50_MEDIAN_REF = 0.008106  # reference only, never used in w180


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def w180(e, theta):
    if not t173(e, theta):
        return 0.0
    return P50_FROZEN / (e["slip_m"] + P50_FROZEN)


def w_alpha_med(e, theta, alpha):
    # reference kernels with R175 median P50 (identity checks only)
    if not t173(e, theta):
        return 0.0
    return (P50_MEDIAN_REF / (e["slip_m"] + P50_MEDIAN_REF)) * (1.0 - alpha * e["friction"])


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
assert round(p50_train, 6) == P50_MEDIAN_REF, "slip P50 drift"
assert abs(P50_FROZEN - P50_MEDIAN_REF) < 2e-4, "director P50 lock too far from median"

# Spot check: manual recompute of w180 for first T173-admitted calib episode
spot = next(e for e in cal_eps if t173(e, theta_marg))
manual = P50_FROZEN / (spot["slip_m"] + P50_FROZEN)
assert abs(w180(spot, theta_marg) - manual) < 1e-15, "weight formula mismatch"
# Identity: w180 vs median-P50 alpha=0 kernel must match within rounding
# (R179 ablation: aggregate p_w delta 0.000; per-episode diff O(3e-3) at small slip)
for e in cal_eps[:5]:
    if t173(e, theta_marg):
        assert abs(w180(e, theta_marg) - w_alpha_med(e, theta_marg, 0.0)) < 1e-2, "P50 lock identity break"

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "n_success_calib": len(sj),
    "P50_frozen_director": P50_FROZEN,
    "P50_train_recomputed": round(p50_train, 6),
    "fric_factor": "none (k=0)",
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0,
               cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "spot_check": {"seed": spot["seed"], "suite": spot["suite"],
                   "slip": spot["slip_m"], "fric": spot["friction"],
                   "w180": round(w180(spot, theta_marg), 6)},
    "trainfold_T173": hard_rule(cal_eps, lambda e: t173(e, theta_marg)),
    "trainfold_T180": soft_eval(cal_eps, theta_marg, w180),
    "held_T173": hard_rule(tst_eps, lambda e: t173(e, theta_marg)),
    "held_T177": soft_eval(tst_eps, theta_marg, lambda e, th: w_alpha_med(e, th, 1.0)),
    "held_T178": soft_eval(tst_eps, theta_marg, lambda e, th: w_alpha_med(e, th, 0.5)),
    "held_T179": soft_eval(tst_eps, theta_marg, lambda e, th: w_alpha_med(e, th, 0.25)),
    "held_T180": soft_eval(tst_eps, theta_marg, w180),
}
h3 = out["held_T173"]
h0 = out["held_T180"]
out["lift_T180_vs_T173_pts"] = round(100 * (h3["p_fail_given_adm"] - h0["p_fail_weighted"]), 2)
out["lift_T180_vs_T177_pts"] = round(100 * (out["held_T177"]["p_fail_weighted"] - h0["p_fail_weighted"]), 2)
out["lift_T180_vs_T178_pts"] = round(100 * (out["held_T178"]["p_fail_weighted"] - h0["p_fail_weighted"]), 2)
out["lift_T180_vs_T179_pts"] = round(100 * (out["held_T179"]["p_fail_weighted"] - h0["p_fail_weighted"]), 2)
out["train_lift_T180_vs_T173_pts"] = round(
    100 * (out["trainfold_T173"]["p_fail_given_adm"] - out["trainfold_T180"]["p_fail_weighted"]), 2)
out["cov_ok"] = h0["cov_w"] >= h0["cov_all"] - 0.02

# Archive valid-mass (LOG-ONLY)
arc_ws = [w180(e, theta_marg) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["archive_T173_valid_rate"] = round(
    sum(1 for e in arc_eps if t173(e, theta_marg)) / len(arc_eps), 4)
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "score_w_ge_70": h0["score_w"] >= 70.0,
    "no_cov_regression": out["cov_ok"],
}
out["keep"] = bool(all(out["verdict_rule"].values()))
# Kill-test diode: fric line closed iff T180 still <= T173 (no signal even at k=0)
out["fric_line_kill"] = out["lift_T180_vs_T173_pts"] < 1.0
json.dump(out, open("results/aegis_v2/I2_r180_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
