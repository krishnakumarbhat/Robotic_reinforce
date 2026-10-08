"""Run 177 offline analysis: T177 pre-reg soft dual-signal shrinker, POST-HOC ONLY.

Director iter 11. Frontier: soft-weighting (replaces failed hard admit/abstain
line T174-T176, closed by R176 frontier-kill). Same signals slip_m + fric +
T173, ZERO hard cuts at P50=0.008106 / fric 0.70.

Frozen inputs (zero rig edits, no rig run): R173 evidence files (20-seed paired,
Run-A keep=True compare record cited, not recomputed).
  CALIB = results/aegis_v2/I2_r173_calib_0012.jsonl   (Run-A decider @0.01,2)
  HELD  = results/aegis_v2/I2_r173_test_0012.jsonl    (Run-B held-out BASE_SEED=190000)
  ARCH  = results/aegis_v2/I2_r173_archive_20.jsonl   (Run-C archive @2.0,2 offline-only)

T173 unified tag (frozen I7+I10): ADMIT iff (jerk<=0.618) AND (jerk<=theta) AND
  (stall_frac<=0.05), theta=0.014133 marginal-conformal 90% quantile over pooled
  Run-A SUCCESS jerks (n=103, alpha=0.1), recomputed LOG-ONLY (bit-identical or
  the rig drifted -> invalid iteration).
T177 rule (PRE-REG, no tuning, single eval, post-hoc forbidden):
  w(e) = 1[T173(e)] * (1 - slip_m/(slip_m+0.008106)) * (1 - fric)
       = 1[T173(e)] * P50/(slip_m+P50) * (1-fric), P50=0.008106 frozen R175
  value (pooled CALIB SUCCESS median; train-only lock: asserted 6dp from CALIB
  here, never refit on held-out). No thresholds, no ablations, no search.
  This is a post-hoc confidence weight, NOT a gate: GATE_MAX_JERK stays 0.618
  frozen, gate_mode stays post-hoc, no veto/control/override (run-169 soft-gate
  class explicitly NOT reopened).

Pre-registered verdict (director: fail iff <70 keep):
  KEEP iff runA_keep (cited) AND score_w >= 70 AND no coverage regression
  (cov_w >= cov_all - 0.02), where score_w = 100 * Prec_w,
  Prec_w = sum(w*succ)/sum(w) on pooled held-out (direct precision analog of
  T173 precision 0.8922; weighted recall is degenerate-by-construction for any
  shrinker and reported for completeness only, never gated).
  Lift = 100*(P(fail|admit)_T173 - P_w(fail)) reported UNGATED (director
  expectation +0.2..0.4pts; negative lift logged as "no evidence of signal").
G7: coverage/success read from physics logs only; weights scale confidence
mass, never per-episode coverage/success (aggregates only, as in R174-R176)."""

import json
import statistics

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
ALPHA = 0.1
I7_THETA = 0.618
STALL_CAP = 0.05
P50_SLIP = 0.008106  # director-frozen R175 value (train-only lock)


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def w177(e, theta):
    if not t173(e, theta):
        return 0.0
    return (P50_SLIP / (e["slip_m"] + P50_SLIP)) * (1.0 - e["friction"])


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


def soft_eval(eps, theta):
    ws = [w177(e, theta) for e in eps]
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

# Train-fold pre-reg locks (LOG-ONLY identity checks; values frozen, never refit)
import math
sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * (1 - ALPHA)) - 1)
theta_marg = sj[k]
assert abs(theta_marg - 0.014133) < 1e-6, f"rig drift: theta={theta_marg}"
succ_slip = sorted(e["slip_m"] for e in cal_eps if e["success"])
assert round(statistics.median(succ_slip), 6) == P50_SLIP, "slip P50 drift"

# Spot check: manual recompute of w for first T173-admitted calib episode
spot = next(e for e in cal_eps if t173(e, theta_marg))
manual = (P50_SLIP / (spot["slip_m"] + P50_SLIP)) * (1.0 - spot["friction"])
assert abs(w177(spot, theta_marg) - manual) < 1e-15, "weight formula mismatch"

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "n_success_calib": len(sj),
    "P50_slip_frozen_trainonly": P50_SLIP,
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0,
               cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "spot_check": {"seed": spot["seed"], "suite": spot["suite"],
                   "slip": spot["slip_m"], "fric": spot["friction"],
                   "w": round(w177(spot, theta_marg), 6)},
    "trainfold_T173": hard_rule(cal_eps, lambda e: t173(e, theta_marg)),
    "trainfold_T177": soft_eval(cal_eps, theta_marg),
    "held_T173": hard_rule(tst_eps, lambda e: t173(e, theta_marg)),
    "held_T177": soft_eval(tst_eps, theta_marg),
}
h3, h7 = out["held_T173"], out["held_T177"]
out["lift_pts_vs_T173"] = round(100 * (h3["p_fail_given_adm"]
                                      - h7["p_fail_weighted"]), 2)
out["cov_ok"] = h7["cov_w"] >= h7["cov_all"] - 0.02
out["no_signal_note"] = (out["lift_pts_vs_T173"] < 0.0)
# Archive valid-mass (LOG-ONLY): soft weights never fully abstain
arc_ws = [w177(e, theta_marg) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["archive_T173_valid_rate"] = round(
    sum(1 for e in arc_eps if t173(e, theta_marg)) / len(arc_eps), 4)
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "score_w_ge_70": h7["score_w"] >= 70.0,
    "no_cov_regression": out["cov_ok"],
}
out["keep"] = bool(all(out["verdict_rule"].values()))
json.dump(out, open("results/aegis_v2/I2_r177_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
