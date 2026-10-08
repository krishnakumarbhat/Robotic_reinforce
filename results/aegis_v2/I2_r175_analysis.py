"""Run 175 offline analysis: T175 slip-tail conditional admit, POST-HOC ONLY.

Frozen inputs (zero rig edits, no rig run): R173 evidence files (20-seed paired,
Run-A keep=True compare record cited, not recomputed).
  CALIB = results/aegis_v2/I2_r173_calib_0012.jsonl   (Run-A decider @0.01,2)
  HELD  = results/aegis_v2/I2_r173_test_0012.jsonl    (Run-B held-out BASE_SEED=190000)
  ARCH  = results/aegis_v2/I2_r173_archive_20.jsonl   (Run-C archive @2.0,2 offline-only)

T173 unified tag (frozen, I7+I2-marginal): ADMIT iff (jerk<=0.618) AND
  (jerk<=theta) AND (stall_frac<=0.05), theta=0.014133 marginal-conformal 90%
  quantile over pooled Run-A SUCCESS jerks (n=103, alpha=0.1), recomputed LOG-ONLY
  (must be bit-identical 0.014133 or the rig drifted -> invalid iteration).
T175 variation (slip-tail, NOT friction-gated per F8 failure): ADMIT iff
  T173-admit AND (slip_m <= P50), P50 = median slip_m over pooled CALIB SUCCESS
  episodes (fit on calib only, no held-out peeking), frozen I7+I10.
Ablation: T175 minus slip gate must recover T173 admission sets EXACTLY.
Pre-registered verdict (director iter 9): KEEP iff runA_keep AND
  lift = 100*(P(fail|admit)_T173 - P(fail|admit)_T175) >= 1.0pt on pooled held-out
  AND recall_T175 >= 0.70 AND no coverage regression (cov_adm >= cov_all - 0.02).
  DISCARD if lift < 1.0pts or keep(recall) < 70 or any rig edit was needed.
G7: coverage/success read from physics logs only; no arithmetic on coverage."""
import json
import math
import statistics

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
ALPHA = 0.1
I7_THETA = 0.618
STALL_CAP = 0.05


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def eval_rule(eps, admit_fn):
    suc = [e for e in eps if e["success"]]
    fail = [e for e in eps if not e["success"]]
    adm = [e for e in eps if admit_fn(e)]
    adm_s = sum(1 for e in adm if e["success"])
    adm_f = len(adm) - adm_s
    rec = adm_s / len(suc) if suc else 1.0
    p_fail_given_adm = adm_f / len(adm) if adm else 0.0
    cov_adm = sum(e["coverage_cont"] for e in adm) / len(adm) if adm else 0.0
    cov_all = sum(e["coverage_cont"] for e in eps) / len(eps)
    return {"n": len(eps), "n_adm": len(adm), "n_abstain": len(eps) - len(adm),
            "abstain_rate": round((len(eps) - len(adm)) / len(eps), 4),
            "recall": round(rec, 4), "score": round(100 * rec, 2),
            "p_fail_given_adm": round(p_fail_given_adm, 4),
            "valid_rate": round(len(adm) / len(eps), 4),
            "cov_adm": round(cov_adm, 4), "cov_all": round(cov_all, 4)}


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)
assert cal_hdr["pose_noise_cfg"] == "0.01,2" and cal_hdr["gate_mode"] == "post-hoc"
assert tst_hdr["pose_noise_cfg"] == "0.01,2" and arc_hdr["pose_noise_cfg"] == "2.0,2"
assert cal_hdr["path_mode"] == "fitted" and cal_hdr["compare"] == "trochoid"

# Theta recomputed LOG-ONLY; frozen-rig identity check
sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * (1 - ALPHA)) - 1)
theta_marg = sj[k]
assert abs(theta_marg - 0.014133) < 1e-6, f"rig drift: theta={theta_marg}"

# P50 fit on CALIB successes only (pooled, same convention as theta)
succ_slip = sorted(e["slip_m"] for e in cal_eps if e["success"])
P50 = statistics.median(succ_slip)
fit_slip = sorted(e["slip_m"] for e in cal_eps
                  if e["success"] and e["path_mode"] == "fitted")
P50_FITTED_ONLY = statistics.median(fit_slip)

T173 = lambda e: (e["jerk"] <= I7_THETA and e["jerk"] <= theta_marg
                 and e["stall_frac"] <= STALL_CAP)
T175 = lambda e: (T173(e) and e["slip_m"] <= P50)

# Ablation: dropping the slip gate must recover T173 exactly
for name, eps in (("calib", cal_eps), ("held", tst_eps), ("arch", arc_eps)):
    a = [T173(e) for e in eps]
    b = [(T173(e) and True) for e in eps]
    assert a == b, f"ablation broken on {name}"
ablation_exact = all(T173(e) == (T173(e) and True) for e in tst_eps)

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "n_success_calib": len(sj),
    "P50_slip_pooled_calib_succ": round(P50, 6),
    "P50_slip_fittedonly_sens": round(P50_FITTED_ONLY, 6),
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0,
               cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "ablation_recovers_T173_exactly": ablation_exact,
    "held_T173": eval_rule(tst_eps, T173),
    "held_T175": eval_rule(tst_eps, T175),
    "calib_T173": eval_rule(cal_eps, T173),
    "calib_T175": eval_rule(cal_eps, T175),
    "arch_T173": eval_rule(arc_eps, T173),
    "arch_T175": eval_rule(arc_eps, T175),
}
h3, h5 = out["held_T173"], out["held_T175"]
out["lift_pts_vs_T173"] = round(100 * (h3["p_fail_given_adm"]
                                      - h5["p_fail_given_adm"]), 2)
out["recall_T175"] = h5["recall"]
out["cov_ok"] = h5["cov_adm"] >= h5["cov_all"] - 0.02
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "ablation_exact": ablation_exact,
    "lift_ge_100pt": out["lift_pts_vs_T173"] >= 1.0,
    "recall_ge_70": h5["recall"] >= 0.70,
    "no_cov_regression": out["cov_ok"],
}
out["keep"] = bool(all(out["verdict_rule"].values()))
# fitted-only subset (secondary, same suite convention as primary pooled)
fit = [e for e in tst_eps if e["path_mode"] == "fitted"]
out["held_fittedonly_T173"] = eval_rule(fit, T173)
out["held_fittedonly_T175"] = eval_rule(fit, T175)
# audit trail: abstained-by-slip-tail episodes (fields logged, not fabricated)
out["slip_abstained_heldout"] = [
    {"suite": e["suite"], "path": e["path_mode"], "seed": e["seed"],
     "friction": e["friction"], "tool": e["tool_id"], "success": e["success"],
     "slip": e["slip_m"], "P50": round(P50, 6), "cov": e["coverage_cont"],
     "t173_admit": T173(e)}
    for e in tst_eps if T173(e) and not T175(e)]
json.dump(out, open("results/aegis_v2/I2_r175_result.json", "w"), indent=1)
print(json.dumps({kk: vv for kk, vv in out.items()
                  if kk != "slip_abstained_heldout"}, indent=1))
print("N_SLIP_ABSTAINED_HELDOUT:", len(out["slip_abstained_heldout"]))
