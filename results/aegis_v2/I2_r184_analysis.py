"""Run 184 offline analysis: T184 fric-free slip-floored damped shrinker, POST-HOC ONLY.

Director iter 9 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: slip-only shrinker, drop fric term entirely (T181-T183 stuck ~1pt;
fric penalty over-shrinks success mass, keep/score < bar).
Variation vs T180/T181/T182/T183: fric-free + stronger slip floor (damped):
  w184(e) = 1[T173(e)] * P50/(max(slip_m,0.005)+P50),
  P50=0.008 director-frozen (no refit),
  no fric term (k=0 dropped entirely, not softened),
  floor 0.005 caps kernel at 0.008/0.013=0.6154 (vs T183 0.8 @0.002,
  T180 uncapped ->1.0, T182 zero below 1e-4).
Vs T180 (P50/(slip+P50)): adds damped floor 0.005 only.
Vs T181 (gated 0.5 fric_n): drops fric + gate entirely.
Vs T182 (0.35 fric_n, zero floor): drops fric, damped not zero floor.
Vs T183 (0.2 fric_n, damp 0.002): drops fric, floor 0.002->0.005.
Pre-reg verdict: KEEP iff runA_keep (cited) AND score_w184 >= 70 AND
  no coverage regression (cov_w >= cov_all - 0.02). Lifts reported UNGATED.
Director FAIL-COND: score_w < 50 -> revert to T183 floor (discard + freeze T183).
Director NEXT: score_w >= 70 -> unfreeze I7+I10 (propose, not execute here).
Frozen inputs (zero rig edits, no rig run): R173 evidence files.
G7: coverage/success from physics logs only; weights scale confidence mass,
  never per-episode coverage/success (aggregates only, as R174-R183)."""

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
SLIP_FLOOR = 0.005  # director spec: max(slip, 0.005) damped floor


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def slip_kernel184(e):
    return P50_FROZEN / (max(e["slip_m"], SLIP_FLOOR) + P50_FROZEN)


def w184(e, theta):
    if not t173(e, theta):
        return 0.0
    return slip_kernel184(e)


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


def w182(e, theta, med):
    if not t173(e, theta):
        return 0.0
    if e["slip_m"] < 1e-4:
        return 0.0
    return (P50_FROZEN / (e["slip_m"] + P50_FROZEN)) * (
        1.0 - 0.35 * min(max(e["friction"] / med, 0.0), 1.0))


def w183(e, theta, med):
    if not t173(e, theta):
        return 0.0
    return (P50_FROZEN / (max(e["slip_m"], 0.002) + P50_FROZEN)) * (
        1.0 - 0.2 * min(e["friction"] / med, 1.0))


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
assert abs(med_fric - 0.3599) < 1e-4, f"med_fric drift: {med_fric}"
assert med_fric > 0, "med_fric degenerate"

# Slip-floor audit (LOG-ONLY): floored eps + kernel cap
floor_hits = {n: sum(1 for e in eps if e["slip_m"] < SLIP_FLOOR)
              for n, eps in [("calib", cal_eps), ("held", tst_eps), ("archive", arc_eps)]}
assert abs(slip_kernel184({"slip_m": 0.0}) - P50_FROZEN / (SLIP_FLOOR + P50_FROZEN)) < 1e-15
assert slip_kernel184({"slip_m": 0.0}) <= 0.6154 + 1e-12, "floor cap broken"

# Spot checks
spot = next(e for e in cal_eps if t173(e, theta_marg))
manual = P50_FROZEN / (max(spot["slip_m"], SLIP_FLOOR) + P50_FROZEN)
assert abs(w184(spot, theta_marg) - manual) < 1e-15, "weight formula mismatch"
for e in cal_eps + tst_eps:
    assert 0.0 <= w184(e, theta_marg) < 1.0 + 1e-12, "w184 out of [0,1)"
    assert ("friction" in e), "friction field missing (frozen check)"

w181fn = lambda e, th: w181(e, th, med_fric)  # noqa: E731
w182fn = lambda e, th: w182(e, th, med_fric)  # noqa: E731
w183fn = lambda e, th: w183(e, th, med_fric)  # noqa: E731

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "n_success_calib": len(sj),
    "P50_frozen_director": P50_FROZEN,
    "P50_train_recomputed": round(p50_train, 6),
    "med_fric_train_success_logonly": round(med_fric, 6),
    "fric_term": "none (dropped entirely)",
    "slip_floor": SLIP_FLOOR,
    "slip_floor_hits": floor_hits,
    "slip_kernel_cap": round(P50_FROZEN / (SLIP_FLOOR + P50_FROZEN), 6),
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0,
               cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "spot_check": {"seed": spot["seed"], "suite": spot["suite"],
                   "slip": spot["slip_m"], "fric": spot["friction"],
                   "w184": round(w184(spot, theta_marg), 6),
                   "w180": round(w180(spot, theta_marg), 6),
                   "w181": round(w181fn(spot, theta_marg), 6),
                   "w182": round(w182fn(spot, theta_marg), 6),
                   "w183": round(w183fn(spot, theta_marg), 6)},
    "trainfold_T173": hard_rule(cal_eps, lambda e: t173(e, theta_marg)),
    "trainfold_T184": soft_eval(cal_eps, theta_marg, w184),
    "held_T173": hard_rule(tst_eps, lambda e: t173(e, theta_marg)),
    "held_T180": soft_eval(tst_eps, theta_marg, w180),
    "held_T181": soft_eval(tst_eps, theta_marg, w181fn),
    "held_T182": soft_eval(tst_eps, theta_marg, w182fn),
    "held_T183": soft_eval(tst_eps, theta_marg, w183fn),
    "held_T184": soft_eval(tst_eps, theta_marg, w184),
}
h3 = out["held_T173"]
h0 = out["held_T180"]
h1 = out["held_T181"]
h2 = out["held_T182"]
h8 = out["held_T183"]
h4 = out["held_T184"]
out["lift_T184_vs_T173_pts"] = round(100 * (h3["p_fail_given_adm"] - h4["p_fail_weighted"]), 2)
out["lift_T184_vs_T180_pts"] = round(100 * (h0["p_fail_weighted"] - h4["p_fail_weighted"]), 2)
out["lift_T184_vs_T181_pts"] = round(100 * (h1["p_fail_weighted"] - h4["p_fail_weighted"]), 2)
out["lift_T184_vs_T182_pts"] = round(100 * (h2["p_fail_weighted"] - h4["p_fail_weighted"]), 2)
out["lift_T184_vs_T183_pts"] = round(100 * (h8["p_fail_weighted"] - h4["p_fail_weighted"]), 2)
out["train_lift_T184_vs_T173_pts"] = round(
    100 * (out["trainfold_T173"]["p_fail_given_adm"] - out["trainfold_T184"]["p_fail_weighted"]), 2)
out["cov_ok"] = h4["cov_w"] >= h4["cov_all"] - 0.02

# Slip histogram on held-out (LOG-ONLY, director validation): 8 bins over observed range
slips = [e["slip_m"] for e in tst_eps]
lo, hi = min(slips), max(slips)
nb = 8
width = (hi - lo) / nb
hist = []
for b in range(nb):
    blo, bhi = lo + b * width, lo + (b + 1) * width
    chunk = [e for e in tst_eps if (blo <= e["slip_m"] < bhi) or (b == nb - 1 and e["slip_m"] == hi)]
    w4 = [w184(e, theta_marg) for e in chunk]
    w0 = [w180(e, theta_marg) for e in chunk]
    hist.append({
        "bin": b,
        "slip_lo": round(blo, 6), "slip_hi": round(bhi, 6),
        "n": len(chunk),
        "n_fail": sum(1 for e in chunk if not e["success"]),
        "mean_w_T184": round(statistics.mean(w4), 4) if w4 else 0.0,
        "mean_w_T180": round(statistics.mean(w0), 4) if w0 else 0.0,
        "fail_w_T184": round(sum(w for w, e in zip(w4, chunk) if not e["success"]), 4),
        "floored": sum(1 for e in chunk if e["slip_m"] < SLIP_FLOOR),
    })
out["slip_hist_held"] = hist

# Slip-decile breakdown on held-out (LOG-ONLY, same episodes for all methods)
order = sorted(tst_eps, key=lambda e: e["slip_m"])
deciles = []
for d in range(10):
    chunk = order[d * 12:(d + 1) * 12]
    w4 = [w184(e, theta_marg) for e in chunk]
    w0 = [w180(e, theta_marg) for e in chunk]
    w1 = [w181fn(e, theta_marg) for e in chunk]
    deciles.append({
        "decile": d,
        "slip_lo": round(min(e["slip_m"] for e in chunk), 6),
        "slip_hi": round(max(e["slip_m"] for e in chunk), 6),
        "n_fail": sum(1 for e in chunk if not e["success"]),
        "mean_w_T180": round(statistics.mean(w0), 4),
        "mean_w_T181": round(statistics.mean(w1), 4),
        "mean_w_T184": round(statistics.mean(w4), 4),
        "fail_w_T180": round(sum(w for w, e in zip(w0, chunk) if not e["success"]), 4),
        "fail_w_T181": round(sum(w for w, e in zip(w1, chunk) if not e["success"]), 4),
        "fail_w_T184": round(sum(w for w, e in zip(w4, chunk) if not e["success"]), 4),
    })
out["slip_deciles_held"] = deciles

# Archive valid-mass (LOG-ONLY)
arc_ws = [w184(e, theta_marg) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["archive_T173_valid_rate"] = round(
    sum(1 for e in arc_eps if t173(e, theta_marg)) / len(arc_eps), 4)
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "score_w_ge_70": h4["score_w"] >= 70.0,
    "no_cov_regression": out["cov_ok"],
}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["fail_cond_score_below_50"] = h4["score_w"] < 50.0
out["head_to_head_win_vs_T180"] = out["lift_T184_vs_T180_pts"] > 0
json.dump(out, open("results/aegis_v2/I2_r184_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
