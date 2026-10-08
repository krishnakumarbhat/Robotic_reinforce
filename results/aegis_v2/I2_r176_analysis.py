"""Run 176 offline analysis: T176 dual-low conditional admit, POST-HOC ONLY.

Frozen inputs (zero rig edits, no rig run): R173 evidence files (20-seed paired,
Run-A keep=True compare record cited, not recomputed).
  CALIB = results/aegis_v2/I2_r173_calib_0012.jsonl   (Run-A decider @0.01,2)
  HELD  = results/aegis_v2/I2_r173_test_0012.jsonl    (Run-B held-out BASE_SEED=190000)
  ARCH  = results/aegis_v2/I2_r173_archive_20.jsonl   (Run-C archive @2.0,2 offline-only)

T173 unified tag (frozen, I7+I2-marginal): ADMIT iff (jerk<=0.618) AND
  (jerk<=theta) AND (stall_frac<=0.05), theta=0.014133 marginal-conformal 90%
  quantile over pooled Run-A SUCCESS jerks (n=103, alpha=0.1), recomputed LOG-ONLY
  (must be bit-identical 0.014133 or the rig drifted -> invalid iteration).
T176 variation (dual-low: 175 slip-tail + inverted-174 friction gate, NO corrector):
  ADMIT iff T173-admit AND (slip_m <= 0.008106) AND (fric < 0.70).
  P50=0.008106 median over pooled CALIB SUCCESS slips (fit calib-only, R175 value);
  fric 0.70 = inverted R174 abstain boundary (abstain iff fric>=0.70).
Ablations: drop-slip (-> T173+fric), drop-fric (-> T175 bit-identity check),
  P75-slip (slip<=P75=0.009804 calib-succ), P75-fric (fric<P75=0.4578 calib-succ).
Pre-registered verdict (director iter 10): KEEP iff runA_keep AND
  lift = 100*(P(fail|admit)_T173 - P(fail|admit)_T176) >= 1.0pt on pooled held-out
  AND recall_T176 >= 0.70 AND no coverage regression (cov_adm >= cov_all - 0.02).
  DISCARD if lift < 1.0pts or keep(recall) < 70. Kill frontier if lift <= 1.0pts:
  close fric/slip hard-threshold line, move to continuous/confidence-weighted tag.
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
P50_SLIP = 0.008106  # director-frozen R175 value
FRIC_LO = 0.70  # inverted R174 boundary: admit iff fric < 0.70


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
    prec = adm_s / len(adm) if adm else 1.0
    p_fail_given_adm = adm_f / len(adm) if adm else 0.0
    cov_adm = sum(e["coverage_cont"] for e in adm) / len(adm) if adm else 0.0
    cov_all = sum(e["coverage_cont"] for e in eps) / len(eps)
    return {"n": len(eps), "n_adm": len(adm), "n_abstain": len(eps) - len(adm),
            "abstain_rate": round((len(eps) - len(adm)) / len(eps), 4),
            "admit_rate": round(len(adm) / len(eps), 4),
            "recall": round(rec, 4), "score": round(100 * rec, 2),
            "precision": round(prec, 4),
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

# P50 recheck LOG-ONLY (must match frozen 0.008106 at 6dp)
succ_slip = sorted(e["slip_m"] for e in cal_eps if e["success"])
assert round(statistics.median(succ_slip), 6) == P50_SLIP, "slip P50 drift"
# P75 values for ablation (fit calib-success only, log-only)
ss = sorted(e["slip_m"] for e in cal_eps if e["success"])
fs = sorted(e["friction"] for e in cal_eps if e["success"])
n = len(ss)
P75_SLIP = (ss[int(0.75 * (n - 1))] + ss[int(0.75 * (n - 1)) + 1]) / 2 \
    if n % 2 == 0 else ss[int(0.75 * n)]
# use linear-interpolated percentile matching numpy default
def pct75(v):
    v = sorted(v)
    pos = 0.75 * (len(v) - 1)
    lo = int(pos)
    return v[lo] + (v[lo + 1] - v[lo]) * (pos - lo)
P75_SLIP = pct75(ss)
P75_FRIC = pct75(fs)

T173 = lambda e: (e["jerk"] <= I7_THETA and e["jerk"] <= theta_marg
                 and e["stall_frac"] <= STALL_CAP)
T175 = lambda e: (T173(e) and e["slip_m"] <= P50_SLIP)
T176 = lambda e: (T173(e) and e["slip_m"] <= P50_SLIP and e["friction"] < FRIC_LO)
# ablations
A_DROP_SLIP = lambda e: (T173(e) and e["friction"] < FRIC_LO)
A_DROP_FRIC = lambda e: (T173(e) and e["slip_m"] <= P50_SLIP)  # == T175
A_P75_SLIP = lambda e: (T173(e) and e["slip_m"] <= P75_SLIP
                        and e["friction"] < FRIC_LO)
A_P75_FRIC = lambda e: (T173(e) and e["slip_m"] <= P50_SLIP
                        and e["friction"] < P75_FRIC)

# Ablation identity: drop-fric must recover T175 exactly
for name, eps in (("calib", cal_eps), ("held", tst_eps), ("arch", arc_eps)):
    assert [A_DROP_FRIC(e) for e in eps] == [T175(e) for e in eps], \
        f"drop-fric != T175 on {name}"
ablation_exact = all(A_DROP_FRIC(e) == T175(e) for e in tst_eps)

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "n_success_calib": len(sj),
    "P50_slip_frozen": P50_SLIP,
    "P75_slip_calib_succ": round(P75_SLIP, 6),
    "P75_fric_calib_succ": round(P75_FRIC, 6),
    "fric_gate": "fric<0.70",
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0,
               cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "ablation_dropfric_is_T175_exactly": ablation_exact,
    "held_T173": eval_rule(tst_eps, T173),
    "held_T175": eval_rule(tst_eps, T175),
    "held_T176": eval_rule(tst_eps, T176),
    "calib_T176": eval_rule(cal_eps, T176),
    "arch_T176": eval_rule(arc_eps, T176),
    "abl_held_drop_slip": eval_rule(tst_eps, A_DROP_SLIP),
    "abl_held_drop_fric": eval_rule(tst_eps, A_DROP_FRIC),
    "abl_held_P75_slip": eval_rule(tst_eps, A_P75_SLIP),
    "abl_held_P75_fric": eval_rule(tst_eps, A_P75_FRIC),
}
h3, h6 = out["held_T173"], out["held_T176"]
out["lift_pts_vs_T173"] = round(100 * (h3["p_fail_given_adm"]
                                      - h6["p_fail_given_adm"]), 2)
out["recall_T176"] = h6["recall"]
out["precision_T176"] = h6["precision"]
out["admit_rate_T176"] = h6["admit_rate"]
out["cov_ok"] = h6["cov_adm"] >= h6["cov_all"] - 0.02
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "ablation_exact": ablation_exact,
    "lift_ge_100pt": out["lift_pts_vs_T173"] >= 1.0,
    "recall_ge_70": h6["recall"] >= 0.70,
    "no_cov_regression": out["cov_ok"],
}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["frontier_kill"] = out["lift_pts_vs_T173"] <= 1.0
# fitted-only subset (secondary)
fit = [e for e in tst_eps if e["path_mode"] == "fitted"]
out["held_fittedonly_T173"] = eval_rule(fit, T173)
out["held_fittedonly_T176"] = eval_rule(fit, T176)
# audit trail: episodes where T176 abstains but T173 admits
out["dual_abstained_heldout"] = [
    {"suite": e["suite"], "path": e["path_mode"], "seed": e["seed"],
     "friction": e["friction"], "tool": e["tool_id"], "success": e["success"],
     "slip": e["slip_m"], "cov": e["coverage_cont"],
     "cut_by_slip": e["slip_m"] > P50_SLIP,
     "cut_by_fric": e["friction"] >= FRIC_LO}
    for e in tst_eps if T173(e) and not T176(e)]
json.dump(out, open("results/aegis_v2/I2_r176_result.json", "w"), indent=1)
print(json.dumps({kk: vv for kk, vv in out.items()
                  if kk != "dual_abstained_heldout"}, indent=1))
print("N_DUAL_ABSTAINED_HELDOUT:", len(out["dual_abstained_heldout"]))
