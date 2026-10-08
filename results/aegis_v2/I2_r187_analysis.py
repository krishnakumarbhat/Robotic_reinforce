"""Run 187 offline analysis: T187 fric-free damped shrinker, T173-only, POST-HOC ONLY.

Director iter 12 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: P-quantile collapse (P50=0 kills all w in P-numerator form).
Variation vs T184/T186: drop P-term, keep damping+floor:
  w187(e) = 1[T173(e)] / (max(slip_m,0.005)+0.02),
  floor 0.005 + damping kept from T184 (beats T185 no-damp ablation),
  NO fric term, I7+I10 frozen, P50 never refit (train median re-asserted log-only).
Validation (director): rank-IC of w, %w=0, keep@70 on same split, no retune.
Pre-reg verdict: KEEP iff runA_keep (cited) AND score_w187 >= 70 AND no cov regression.
Branch: lift>=+10pts vs T173-hard? P-collapse confirmed -> next add fric-eps to
  restore P50>0. lift~0 (scale-proportional to T186) -> kill fric-free line, unfreeze I10.
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
P50_EFF = 0.02
SLIP_FLOOR = 0.005
DAMP_OFF = 0.02  # == P50_eff denominator offset, P numerator dropped


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def w187(e, theta):
    return 1.0 / (max(e["slip_m"], SLIP_FLOOR) + DAMP_OFF) if t173(e, theta) else 0.0


def w186(e, theta):
    return P50_EFF / (max(e["slip_m"], SLIP_FLOOR) + P50_EFF) if t173(e, theta) else 0.0


def w184frozen(e, theta):
    return P50_FROZEN / (max(e["slip_m"], SLIP_FLOOR) + P50_FROZEN) if t173(e, theta) else 0.0


def w180(e, theta):
    if not t173(e, theta):
        return 0.0
    return P50_FROZEN / (e["slip_m"] + P50_FROZEN)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    n = len(xs)
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


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
    assert all(w >= 0.0 for w in ws)
    assert all((w == 0.0) == (not t173(e, theta)) for w, e in zip(ws, eps)), "T173 support mismatch"
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
    ic_succ = spearman(ws, [1.0 if e["success"] else 0.0 for e in eps])
    ic_cov = spearman(ws, [e["coverage_cont"] for e in eps])
    return {"n": len(eps), "sum_w": round(sw, 4),
            "mean_w": round(sw / len(eps), 4),
            "ess": round(ess, 2),
            "p_fail_weighted": round(p_w, 4),
            "precision_w": round(prec_w, 4),
            "score_w": round(100 * prec_w, 2),
            "recall_w": round(rec_w, 4),
            "cov_w": round(cov_w, 4), "cov_all": round(cov_all, 4),
            "pct_w_zero": round(sum(1 for w in ws if w == 0.0) / len(ws), 4),
            "rankIC_w_vs_success": round(ic_succ, 4),
            "rankIC_w_vs_coverage": round(ic_cov, 4)}


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
assert P50_EFF == 0.02 and DAMP_OFF == 0.02

# P-collapse demo: P-numerator form with P50=0 -> all-zero weights (kills all w).
w_pzero = [0.0 / (max(e["slip_m"], SLIP_FLOOR) + 0.0) if t173(e, theta_marg) else 0.0
           for e in tst_eps]
assert all(w == 0.0 for w in w_pzero), "P50=0 must kill all w in numerator form"

# Scale-proportionality: w187 == w186 / P50_eff exactly (same support, same ranking).
for e in cal_eps + tst_eps + arc_eps:
    a, b = w187(e, theta_marg), w186(e, theta_marg)
    assert (a == 0.0) == (b == 0.0)
    if a > 0:
        assert abs(a * P50_EFF - b) < 1e-9, (a, b)

floor_hits = {n: sum(1 for e in eps if e["slip_m"] < SLIP_FLOOR)
              for n, eps in [("calib", cal_eps), ("held", tst_eps), ("archive", arc_eps)]}
assert abs(1.0 / (SLIP_FLOOR + DAMP_OFF) - 40.0) < 1e-9  # unnormalized cap
for e in cal_eps + tst_eps:
    assert 0.0 <= w187(e, theta_marg) <= 40.0 + 1e-9

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "n_success_calib": len(sj),
    "P50_frozen_director": P50_FROZEN,
    "P50_train_recomputed": round(p50_train, 6),
    "P50_eff": P50_EFF,
    "variant": "T187 fric-free damped shrinker w=T173/(max(slip,0.005)+0.02), P-term dropped",
    "fric_term": "none",
    "slip_floor": SLIP_FLOOR,
    "slip_floor_hits": floor_hits,
    "w_cap_unnormalized": 40.0,
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "p_collapse_demo": {"P50_zero_form_pct_w_zero_held": 1.0,
                        "T187_pct_w_zero_held": round(sum(1 for e in tst_eps if not t173(e, theta_marg)) / len(tst_eps), 4)},
    "trainfold_T173": hard_rule(cal_eps, lambda e: t173(e, theta_marg)),
    "trainfold_T187": soft_eval(cal_eps, theta_marg, w187),
    "held_T173": hard_rule(tst_eps, lambda e: t173(e, theta_marg)),
    "held_T180": soft_eval(tst_eps, theta_marg, w180),
    "held_T184frozen": soft_eval(tst_eps, theta_marg, w184frozen),
    "held_T186": soft_eval(tst_eps, theta_marg, w186),
    "held_T187": soft_eval(tst_eps, theta_marg, w187),
}
h3, h0, hr, h6, h7 = out["held_T173"], out["held_T180"], out["held_T184frozen"], out["held_T186"], out["held_T187"]
out["lift_T187_vs_T173_pts"] = round(100 * (h3["p_fail_given_adm"] - h7["p_fail_weighted"]), 2)
out["lift_T187_vs_T180_pts"] = round(100 * (h0["p_fail_weighted"] - h7["p_fail_weighted"]), 2)
out["lift_T187_vs_T184_pts"] = round(100 * (hr["p_fail_weighted"] - h7["p_fail_weighted"]), 2)
out["lift_T187_vs_T186_pts"] = round(100 * (h6["p_fail_weighted"] - h7["p_fail_weighted"]), 2)
out["train_lift_T187_vs_T173_pts"] = round(
    100 * (out["trainfold_T173"]["p_fail_given_adm"] - out["trainfold_T187"]["p_fail_weighted"]), 2)
out["scale_proportional_to_T186"] = (
    out["lift_T187_vs_T186_pts"] == 0.0 and h7["score_w"] == h6["score_w"]
    and h7["rankIC_w_vs_success"] == h6["rankIC_w_vs_success"])
out["cov_ok"] = h7["cov_w"] >= h7["cov_all"] - 0.02
arc_ws = [w187(e, theta_marg) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["archive_T173_valid_rate"] = round(sum(1 for e in arc_eps if t173(e, theta_marg)) / len(arc_eps), 4)
out["verdict_rule"] = {"runA_keep": out["runA_keep_cited"],
                       "score_w_ge_70": h7["score_w"] >= 70.0,
                       "no_cov_regression": out["cov_ok"]}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["director_branch"] = ("P-collapse-confirmed-scale-only"
                           if out["scale_proportional_to_T186"] else "needs-review")
json.dump(out, open("results/aegis_v2/I2_r187_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
