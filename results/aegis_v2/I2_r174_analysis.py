"""Run 174 offline analysis: F8 friction-binned pitch-residual corrector T174, POST-HOC ONLY.

Frozen inputs (zero rig edits, no rig run): R173 evidence files (20-seed paired,
Run-A keep=True compare record cited, not recomputed).
  CALIB = results/aegis_v2/I2_r173_calib_0012.jsonl   (Run-A decider @0.01,2)
  HELD  = results/aegis_v2/I2_r173_test_0012.jsonl    (Run-B held-out BASE_SEED=190000)
  ARCH  = results/aegis_v2/I2_r173_archive_20.jsonl   (Run-C archive @2.0,2 offline-only)

T174 (score weight only, never control): ABSTAIN iff (friction >= 0.70) AND
  (slip_m - rim_margin > 0), where rim_margin is ANALYTIC from the frozen I10 rule
  (no fitting): side = 0.12 elongated else 0.18; r_eff = {0:0.035,1:0.04,2:0.05};
  n = ceil(side/2r); pitch = side/n; margin = r_eff - pitch/2.
  Variation vs T173: I7 conjunct DROPPED (0 lift 172->173, verify binds=0);
  I2-marginal theta recomputed but LOG-ONLY (recorded, not enforced).
Pre-registered verdict: KEEP iff runA_keep AND recall>=0.70 AND P(fail|admit)<=0.15
  AND lift = 100*(P(fail|admit)_OFF - P(fail|admit)_ON) >= 1.0pt on pooled held-out
  AND no coverage regression (cov_adm >= cov_all - 0.02) AND archive valid rate 0.0.
G7: coverage/success read from physics logs only; no arithmetic on coverage."""
import json
import math

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
R_EFF = {0: 0.035, 1: 0.04, 2: 0.05}
FRIC_HI = 0.70
ALPHA = 0.1


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def rim_margin(e):
    shape = (e.get("fixture_spec") or {}).get("tank_shape", "round")
    side = 0.12 if shape == "elongated" else 0.18
    r = R_EFF[int(e["tool_id"]) % 3]
    n = max(1, int(math.ceil(side / (2.0 * r))))
    pitch = side / n
    return round(r - pitch / 2.0, 6)


def t174_abstain(e):
    return bool(e["friction"] >= FRIC_HI and (e["slip_m"] - rim_margin(e)) > 0)


def i7_binds(e):
    return not (e["jerk"] <= 0.618)


def eval_rule(eps, admit_fn):
    suc = [e for e in eps if e["success"]]
    fail = [e for e in eps if not e["success"]]
    adm = [e for e in eps if admit_fn(e)]
    adm_s = sum(1 for e in adm if e["success"])
    adm_f = len(adm) - adm_s
    rec = adm_s / len(suc) if suc else 1.0
    p_adm_given_fail = adm_f / len(fail) if fail else 0.0
    p_fail_given_adm = adm_f / len(adm) if adm else 0.0
    cov_adm = sum(e["coverage_cont"] for e in adm) / len(adm) if adm else 0.0
    cov_all = sum(e["coverage_cont"] for e in eps) / len(eps)
    return {"n": len(eps), "n_adm": len(adm),
            "recall": round(rec, 4), "score": round(100 * rec, 2),
            "p_adm_given_fail": round(p_adm_given_fail, 4),
            "p_fail_given_adm": round(p_fail_given_adm, 4),
            "valid_rate": round(len(adm) / len(eps), 4),
            "cov_adm": round(cov_adm, 4), "cov_all": round(cov_all, 4)}


def bin_split(eps, admit_fn):
    out = {}
    for name, lo, hi in [("lo[0.05,0.30)", 0.05, 0.30), ("mid[0.30,0.55)", 0.30, 0.55),
                         ("hi[0.55,0.70)", 0.55, 0.70), ("top[0.70,0.80]", 0.70, 0.81)]:
        sub = [e for e in eps if lo <= e["friction"] < hi]
        s = sum(1 for e in sub if e["success"])
        a = sum(1 for e in sub if admit_fn(e))
        af = sum(1 for e in sub if admit_fn(e) and not e["success"])
        covs = [e["coverage_cont"] for e in sub]
        out[name] = {"n": len(sub), "succ": s,
                     "mincov": round(min(covs), 4) if covs else None,
                     "n_abstain": len(sub) - a,
                     "abstained_fail": sum(1 for e in sub if not admit_fn(e) and not e["success"]),
                     "abstained_succ": sum(1 for e in sub if not admit_fn(e) and e["success"]),
                     "p_fail_given_adm": round(af / a, 4) if a else None}
    return out


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)
assert cal_hdr["pose_noise_cfg"] == "0.01,2" and cal_hdr["gate_mode"] == "post-hoc"
assert tst_hdr["pose_noise_cfg"] == "0.01,2" and arc_hdr["pose_noise_cfg"] == "2.0,2"
assert cal_hdr["path_mode"] == "fitted" and cal_hdr["compare"] == "trochoid"

# I2-marginal theta recomputed, LOG-ONLY
sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * (1 - ALPHA)) - 1)
theta_marg = sj[k]

OFF = lambda e: True
ON = lambda e: not t174_abstain(e)

out = {
    "theta_marginal_logonly": round(theta_marg, 6), "n_success_calib": len(sj),
    "fric_hi_bin": FRIC_HI,
    "margins_analytic": {"B_t0": 0.005, "B_t1": 0.01, "B_t2": 0.02,
                         "A_t0": 0.005, "A_t1": 0.01, "A_t2": 0.005},
    "i7_binds_heldout": sum(1 for e in tst_eps if i7_binds(e)),
    "i7_binds_calib": sum(1 for e in cal_eps if i7_binds(e)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "calib_OFF": eval_rule(cal_eps, OFF), "calib_ON": eval_rule(cal_eps, ON),
    "held_OFF": eval_rule(tst_eps, OFF), "held_ON": eval_rule(tst_eps, ON),
    "arch_ON": eval_rule(arc_eps, ON),
    "held_ON_bins": bin_split(tst_eps, ON),
    "held_OFF_bins": bin_split(tst_eps, OFF),
    "calib_ON_bins": bin_split(cal_eps, ON),
}
h_on, h_off = out["held_ON"], out["held_OFF"]
out["lift_pts"] = round(100 * (h_off["p_fail_given_adm"] - h_on["p_fail_given_adm"]), 2)
out["recall_ON"] = h_on["recall"]
out["cov_ok"] = h_on["cov_adm"] >= h_on["cov_all"] - 0.02
out["arch_valid_rate"] = out["arch_ON"]["valid_rate"]
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "recall_ge_70": h_on["recall"] >= 0.70,
    "p_fail_given_adm_le_15": h_on["p_fail_given_adm"] <= 0.15,
    "lift_ge_100pt": out["lift_pts"] >= 1.0,
    "no_cov_regression": out["cov_ok"],
    "archive_abstention": out["arch_valid_rate"] == 0.0,
}
out["keep"] = bool(all(out["verdict_rule"].values()))
# abstention detail (audit trail: which episodes, all fields logged not fabricated)
out["abstained_heldout"] = [
    {"suite": e["suite"], "path": e["path_mode"], "seed": e["seed"],
     "friction": e["friction"], "tool": e["tool_id"], "success": e["success"],
     "slip": e["slip_m"], "margin": rim_margin(e),
     "resid": round(e["slip_m"] - rim_margin(e), 5), "cov": e["coverage_cont"]}
    for e in tst_eps if t174_abstain(e)]
out["abstained_calib"] = [
    {"suite": e["suite"], "path": e["path_mode"], "seed": e["seed"],
     "friction": e["friction"], "tool": e["tool_id"], "success": e["success"],
     "slip": e["slip_m"], "margin": rim_margin(e),
     "resid": round(e["slip_m"] - rim_margin(e), 5), "cov": e["coverage_cont"]}
    for e in cal_eps if t174_abstain(e)]
json.dump(out, open("/tmp/i174_result.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("abstained_heldout", "abstained_calib")}, indent=1))
print("ABSTAINED-HELDOUT:", json.dumps(out["abstained_heldout"], indent=1))
print("ABSTAINED-CALIB:", json.dumps(out["abstained_calib"], indent=1))
