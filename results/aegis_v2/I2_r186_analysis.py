"""Run 186 offline analysis: T186 fric-free damped shrinker with P50_eff fix, POST-HOC ONLY.

Director iter 11 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: zero-median weight degeneracy, not friction/damping.
Variation vs T184 (identical except weight fix):
  w186(e) = 1[T173(e)] * P50_eff/(max(slip_m,0.005)+P50_eff),
  P50=0.008 director-frozen (train median 0.008106 re-asserted, never refit),
  P50_eff = max(P50, 0.02) = 0.02,
  slip floor 0.005 (same as T184, damped on), NO fric term, I7+I10 frozen.
Hold constant: floor, damping, no-fric, no unfreeze — isolate one fix.
Validation: replay T183/T184/T185 kernels with P50_eff only (monotonic w>0 check);
  lift vs T173 hard tag reported post-hoc (1.0-baseline = weight-1.0 hard tag).
Kill rule: if T186 still ~T173 (lift ~0), kill shrinker line, audit T173 upstream.
Pre-reg verdict: KEEP iff runA_keep (cited) AND score_w186 >= 70 AND no cov regression.
Frozen inputs (zero rig edits, no rig run): R173 evidence files.
G7: weights scale confidence mass only, never per-episode coverage/success."""

import json
import math
import statistics

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
ALPHA = 0.1
I7_THETA = 0.618
STALL_CAP = 0.05
P50_FROZEN = 0.008
P50_EFF = max(P50_FROZEN, 0.02)
SLIP_FLOOR = 0.005


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def kern(p50eff, slip):
    return p50eff / (max(slip, SLIP_FLOOR) + p50eff)


def w186(e, theta):
    return kern(P50_EFF, e["slip_m"]) if t173(e, theta) else 0.0


def w_t184(e, theta):
    return kern(P50_FROZEN, e["slip_m"]) if t173(e, theta) else 0.0


def w180(e, theta):
    if not t173(e, theta):
        return 0.0
    return P50_FROZEN / (e["slip_m"] + P50_FROZEN)


def w181(e, theta, med):
    if not t173(e, theta):
        return 0.0
    gate = 1.0 if e["slip_m"] < P50_FROZEN else 0.0
    return (P50_FROZEN / (e["slip_m"] + P50_FROZEN)) * (
        1.0 - 0.5 * min(max(e["friction"] / med, 0.0), 1.0) * gate)


def w183(e, theta, med):
    if not t173(e, theta):
        return 0.0
    return (P50_FROZEN / (max(e["slip_m"], 0.002) + P50_FROZEN)) * (
        1.0 - 0.2 * min(e["friction"] / med, 1.0))


def w185(e, theta):
    if not t173(e, theta):
        return 0.0
    return P50_FROZEN / (max(e["slip_m"], 0.01) + P50_FROZEN)


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
    assert all(0.0 <= w < 1.0 + 1e-12 for w in ws)
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
assert cal_hdr["path_mode"] == "fitted"

sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * (1 - ALPHA)) - 1)
theta_marg = sj[k]
assert abs(theta_marg - 0.014133) < 1e-6, f"rig drift: theta={theta_marg}"
succ_slip = sorted(e["slip_m"] for e in cal_eps if e["success"])
p50_train = statistics.median(succ_slip)
assert abs(P50_FROZEN - p50_train) < 2e-4
succ_fric = sorted(e["friction"] for e in cal_eps if e["success"])
med_fric = statistics.median(succ_fric)
assert abs(med_fric - 0.3599) < 1e-4
assert P50_EFF == 0.02

# Director validation 1: replay T183/T184/T185 kernels with P50_eff only.
# Monotonic decreasing in slip, strictly > 0 on T173 support.
grid = [0.0, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05]
rep = {}
for tag, fn in [("T183", lambda s: P50_EFF / (max(s, 0.002) + P50_EFF)),
                ("T184", lambda s: P50_EFF / (max(s, 0.005) + P50_EFF)),
                ("T185", lambda s: P50_EFF / (max(s, 0.01) + P50_EFF)),
                ("T186", lambda s: P50_EFF / (max(s, 0.005) + P50_EFF))]:
    ws = [fn(s) for s in grid]
    rep[tag] = {"grid": grid, "w": [round(w, 6) for w in ws],
                "monotonic_nonincreasing": all(a >= b - 1e-12 for a, b in zip(ws, ws[1:])),
                "strictly_positive": all(w > 0 for w in ws)}
assert all(v["monotonic_nonincreasing"] and v["strictly_positive"] for v in rep.values()), rep
assert abs(kern(P50_EFF, 0.0) - 0.02 / 0.025) < 1e-15
assert abs(kern(P50_EFF, 0.0) - 0.8) < 1e-12

floor_hits = {n: sum(1 for e in eps if e["slip_m"] < SLIP_FLOOR)
              for n, eps in [("calib", cal_eps), ("held", tst_eps), ("archive", arc_eps)]}
spot = next(e for e in cal_eps if t173(e, theta_marg))
assert abs(w186(spot, theta_marg) - kern(P50_EFF, spot["slip_m"])) < 1e-15
for e in cal_eps + tst_eps:
    assert 0.0 <= w186(e, theta_marg) < 1.0 + 1e-12

w181fn = lambda e, th: w181(e, th, med_fric)  # noqa: E731
w183fn = lambda e, th: w183(e, th, med_fric)  # noqa: E731

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "n_success_calib": len(sj),
    "P50_frozen_director": P50_FROZEN,
    "P50_train_recomputed": round(p50_train, 6),
    "P50_eff": P50_EFF,
    "med_fric_train_success_logonly": round(med_fric, 6),
    "fric_term": "none (dropped entirely, identical to T184)",
    "slip_floor": SLIP_FLOOR,
    "slip_floor_hits": floor_hits,
    "slip_kernel_cap_eff": round(P50_EFF / (SLIP_FLOOR + P50_EFF), 6),
    "slip_kernel_cap_frozen": round(P50_FROZEN / (SLIP_FLOOR + P50_FROZEN), 6),
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0,
               cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "replay_P50eff_only": rep,
    "spot_check": {"seed": spot["seed"], "suite": spot["suite"],
                   "slip": spot["slip_m"], "fric": spot["friction"],
                   "w186": round(w186(spot, theta_marg), 6),
                   "w184frozen": round(w_t184(spot, theta_marg), 6)},
    "trainfold_T173": hard_rule(cal_eps, lambda e: t173(e, theta_marg)),
    "trainfold_T186": soft_eval(cal_eps, theta_marg, w186),
    "held_T173": hard_rule(tst_eps, lambda e: t173(e, theta_marg)),
    "held_T180": soft_eval(tst_eps, theta_marg, w180),
    "held_T181": soft_eval(tst_eps, theta_marg, w181fn),
    "held_T183": soft_eval(tst_eps, theta_marg, w183fn),
    "held_T184frozen": soft_eval(tst_eps, theta_marg, w_t184),
    "held_T185": soft_eval(tst_eps, theta_marg, w185),
    "held_T186": soft_eval(tst_eps, theta_marg, w186),
}
h3 = out["held_T173"]
h0 = out["held_T180"]
h1 = out["held_T181"]
h8 = out["held_T183"]
hr = out["held_T184frozen"]
h5 = out["held_T185"]
h6 = out["held_T186"]
out["lift_T186_vs_T173_pts"] = round(100 * (h3["p_fail_given_adm"] - h6["p_fail_weighted"]), 2)
out["lift_T186_vs_T180_pts"] = round(100 * (h0["p_fail_weighted"] - h6["p_fail_weighted"]), 2)
out["lift_T186_vs_T181_pts"] = round(100 * (h1["p_fail_weighted"] - h6["p_fail_weighted"]), 2)
out["lift_T186_vs_T183_pts"] = round(100 * (h8["p_fail_weighted"] - h6["p_fail_weighted"]), 2)
out["lift_T186_vs_T184_pts"] = round(100 * (hr["p_fail_weighted"] - h6["p_fail_weighted"]), 2)
out["lift_T186_vs_T185_pts"] = round(100 * (h5["p_fail_weighted"] - h6["p_fail_weighted"]), 2)
out["train_lift_T186_vs_T173_pts"] = round(
    100 * (out["trainfold_T173"]["p_fail_given_adm"] - out["trainfold_T186"]["p_fail_weighted"]), 2)
out["cov_ok"] = h6["cov_w"] >= h6["cov_all"] - 0.02
w6 = [w186(e, theta_marg) for e in tst_eps]
w4 = [w_t184(e, theta_marg) for e in tst_eps]
out["keepdist_vs_T184"] = {
    "mean_w_T186": h6["mean_w"], "mean_w_T184": hr["mean_w"],
    "mean_w_delta": round(h6["mean_w"] - hr["mean_w"], 4),
    "ess_T186": h6["ess"], "ess_T184": hr["ess"],
    "recall_T186": h6["recall_w"], "recall_T184": hr["recall_w"],
    "recall_delta": round(h6["recall_w"] - hr["recall_w"], 4),
    "score_T186": h6["score_w"], "score_T184": hr["score_w"],
    "mean_abs_wdiff": round(statistics.mean(abs(a - b) for a, b in zip(w6, w4)), 6),
    "max_abs_wdiff": round(max(abs(a - b) for a, b in zip(w6, w4)), 6),
}
arc_ws = [w186(e, theta_marg) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["archive_T173_valid_rate"] = round(
    sum(1 for e in arc_eps if t173(e, theta_marg)) / len(arc_eps), 4)
out["kill_check_shrinker_vs_10"] = abs(out["lift_T186_vs_T173_pts"]) < 0.05
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "score_w_ge_70": h6["score_w"] >= 70.0,
    "no_cov_regression": out["cov_ok"],
}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["fail_cond_score_below_50"] = h6["score_w"] < 50.0
out["head_to_head_win_vs_T184"] = out["lift_T186_vs_T184_pts"] > 0
out["head_to_head_win_vs_T180"] = out["lift_T186_vs_T180_pts"] > 0
json.dump(out, open("results/aegis_v2/I2_r186_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
