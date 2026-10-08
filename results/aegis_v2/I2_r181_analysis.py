"""Run 181 offline analysis: T181 slip-gated normalized-fric shrinker, POST-HOC ONLY.

Director iter 15 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: fric-CALIBRATION, not fric-slope (k=0.5/0.25/0 sweep exhausted R177-R180).
Variation vs T180 zero-fric control:
  w181(e) = 1[T173(e)] * P50/(slip_m+P50) * (1 - 0.5*fric_n*gate),
  P50=0.008 director-frozen (no refit),
  fric_n = clip(fric/med_fric, 0, 1), med_fric = median friction over
    TRAIN(calib)-fold SUCCESS episodes only (train-locked calibration),
  gate = 1 iff slip_m < P50 else 0 (slip-gated: fric penalty applies only in
    low-slip regime; kills false fric penalties in high-slip regime).
Why: T177-T180 showed raw-fric penalty monotonically harmful (+0.16pts per
  +0.25 alpha); hypothesis: raw fric penalizes high-fric successes that the
  slip kernel already downweights correctly. Normalization bounds the penalty
  to [0,0.5]; gating removes it where slip already signals.
Pre-reg verdict: KEEP iff runA_keep (cited) AND score_w181 >= 70 AND
  no coverage regression (cov_w >= cov_all - 0.02). Lifts reported UNGATED.
Kill (director): if keep<70 (score_w) OR held P_w(fail) <= T180 (no head-to-head
  win vs zero-fric control), kill fric-penalty line, pivot to P50 sweep on T180.
Frozen inputs (zero rig edits, no rig run): R173 evidence files.
G7: coverage/success from physics logs only; weights scale confidence mass,
  never per-episode coverage/success (aggregates only, as R174-R180)."""

import json
import math
import statistics

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
ALPHA = 0.1
I7_THETA = 0.618
STALL_CAP = 0.05
P50_FROZEN = 0.008  # director lock (cf R175 median 0.008106)
FRIC_COEF = 0.5  # director spec


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def fric_n(e, med):
    return min(max(e["friction"] / med, 0.0), 1.0)


def gate(e):
    return 1.0 if e["slip_m"] < P50_FROZEN else 0.0


def w181(e, theta, med):
    if not t173(e, theta):
        return 0.0
    return (P50_FROZEN / (e["slip_m"] + P50_FROZEN)) * (
        1.0 - FRIC_COEF * fric_n(e, med) * gate(e))


def w180(e, theta):
    if not t173(e, theta):
        return 0.0
    return P50_FROZEN / (e["slip_m"] + P50_FROZEN)


def w181_gate_on(e, theta, med):  # ablation: gate forced 1
    if not t173(e, theta):
        return 0.0
    return (P50_FROZEN / (e["slip_m"] + P50_FROZEN)) * (
        1.0 - FRIC_COEF * fric_n(e, med))


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
assert abs(P50_FROZEN - p50_train) < 2e-4, "director P50 lock too far from median"
succ_fric = sorted(e["friction"] for e in cal_eps if e["success"])
med_fric = statistics.median(succ_fric)
assert med_fric > 0, "med_fric degenerate"

# Spot checks
spot = next(e for e in cal_eps if t173(e, theta_marg))
manual = (P50_FROZEN / (spot["slip_m"] + P50_FROZEN)) * (
    1.0 - FRIC_COEF * fric_n(spot, med_fric) * gate(spot))
assert abs(w181(spot, theta_marg, med_fric) - manual) < 1e-15, "weight formula mismatch"
# Identity: gate-off episodes must equal T180 exactly
for e in cal_eps:
    if t173(e, theta_marg) and gate(e) == 0.0:
        assert abs(w181(e, theta_marg, med_fric) - w180(e, theta_marg)) < 1e-15, \
            "gate-off identity break (must equal T180)"
# fric_n bounds
for e in cal_eps + tst_eps:
    assert 0.0 <= fric_n(e, med_fric) <= 1.0, "fric_n out of [0,1]"

w181fn = lambda e, th: w181(e, th, med_fric)
wgon = lambda e, th: w181_gate_on(e, th, med_fric)

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "n_success_calib": len(sj),
    "P50_frozen_director": P50_FROZEN,
    "P50_train_recomputed": round(p50_train, 6),
    "med_fric_train_success": round(med_fric, 6),
    "fric_coef": FRIC_COEF,
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0,
               cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "gate_rate_held": round(sum(gate(e) for e in tst_eps if t173(e, theta_marg)) /
                            max(1, sum(1 for e in tst_eps if t173(e, theta_marg))), 4),
    "gate_rate_calib": round(sum(gate(e) for e in cal_eps if t173(e, theta_marg)) /
                             max(1, sum(1 for e in cal_eps if t173(e, theta_marg))), 4),
    "spot_check": {"seed": spot["seed"], "suite": spot["suite"],
                   "slip": spot["slip_m"], "fric": spot["friction"],
                   "fric_n": round(fric_n(spot, med_fric), 6),
                   "gate": gate(spot),
                   "w181": round(w181(spot, theta_marg, med_fric), 6),
                   "w180": round(w180(spot, theta_marg), 6)},
    "trainfold_T173": hard_rule(cal_eps, lambda e: t173(e, theta_marg)),
    "trainfold_T181": soft_eval(cal_eps, theta_marg, w181fn),
    "trainfold_T181_gateon": soft_eval(cal_eps, theta_marg, wgon),
    "held_T173": hard_rule(tst_eps, lambda e: t173(e, theta_marg)),
    "held_T180": soft_eval(tst_eps, theta_marg, w180),
    "held_T181": soft_eval(tst_eps, theta_marg, w181fn),
    "held_T181_gateon": soft_eval(tst_eps, theta_marg, wgon),
}
h3 = out["held_T173"]
h0 = out["held_T180"]
h1 = out["held_T181"]
hg = out["held_T181_gateon"]
out["lift_T181_vs_T173_pts"] = round(100 * (h3["p_fail_given_adm"] - h1["p_fail_weighted"]), 2)
out["lift_T181_vs_T180_pts"] = round(100 * (h0["p_fail_weighted"] - h1["p_fail_weighted"]), 2)
out["lift_gateon_vs_T180_pts"] = round(100 * (h0["p_fail_weighted"] - hg["p_fail_weighted"]), 2)
out["lift_gate_effect_pts"] = round(100 * (hg["p_fail_weighted"] - h1["p_fail_weighted"]), 2)
out["train_lift_T181_vs_T173_pts"] = round(
    100 * (out["trainfold_T173"]["p_fail_given_adm"] - out["trainfold_T181"]["p_fail_weighted"]), 2)
out["cov_ok"] = h1["cov_w"] >= h1["cov_all"] - 0.02

# Archive valid-mass (LOG-ONLY)
arc_ws = [w181fn(e, theta_marg) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["archive_T173_valid_rate"] = round(
    sum(1 for e in arc_eps if t173(e, theta_marg)) / len(arc_eps), 4)
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "score_w_ge_70": h1["score_w"] >= 70.0,
    "no_cov_regression": out["cov_ok"],
}
out["keep"] = bool(all(out["verdict_rule"].values()))
# Director kill: fric-penalty line dead iff keep fails OR T181 <= T180 head-to-head
out["head_to_head_win_vs_T180"] = out["lift_T181_vs_T180_pts"] > 0
out["fric_line_kill"] = (not out["keep"]) or (not out["head_to_head_win_vs_T180"])
out["pivot"] = "P50 sweep on T180" if out["fric_line_kill"] else "none"
json.dump(out, open("results/aegis_v2/I2_r181_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
