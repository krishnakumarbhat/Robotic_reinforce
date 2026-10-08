"""Run 189 offline analysis: T189 = T188 ablation, damping OFF, POST-HOC ONLY.

Director iter 14 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: damped-shrinker is culprit, not eps/slip floor.
Variation vs T188: keep w=T173/(max(slip,0.005)+eps*) form BUT damping OFF
  (remove max() floor -> raw slip), eps*=0.01 = pre-reg tuned best from 188
  (eps_rule max(median_calib(slip>0),0.01) train-locked; sweep-observed best
  0.05 reported log-only, NOT refit); shrink lambda 1.0->0 arm = pure T173
  hard tag (denominator shrinker removed entirely) cited via held_T173.
Validation (director): same post-hoc slip replay on frozen R173 files;
  check turnover (mean|max|dw| T189-vs-T188), breadth (ESS/mean_w/pct_zero),
  P50 hit-rate (score_w/recall_w/precision_w) vs T188.
Pre-reg verdict: KEEP iff runA_keep (cited) AND score_w189 >= 70 AND no cov
  regression. Predicted 20-35 vs 1.0 baseline -> keep if score_w delta > 10
  (director scale); fail path (still ~T188): kill shrinker family, propose
  iter-15 unfreeze I10 only (not executed here).
Frozen inputs (zero rig edits, no rig run, no live, no refit): R173 files.
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
SLIP_FLOOR = 0.005
EPS_STAR = 0.01  # pre-reg tuned best from 188 (train-locked)


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def w188(e, theta):
    return 1.0 / (max(e["slip_m"], SLIP_FLOOR) + EPS_STAR) if t173(e, theta) else 0.0


def w189(e, theta):
    # damping OFF: raw slip, no floor
    return 1.0 / (e["slip_m"] + EPS_STAR) if t173(e, theta) else 0.0


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
calib_pos = [e["slip_m"] for e in cal_eps if e["slip_m"] > 0]
med_pos = statistics.median(calib_pos)
assert max(med_pos, 0.01) == EPS_STAR == 0.01, f"eps*={EPS_STAR} med_pos={med_pos}"

# Bit-match frozen T188 tuned arm from r188 result file.
w188ref = json.load(open("results/aegis_v2/I2_r188_result.json"))["held_T188_tuned"]
chk = soft_eval(tst_eps, theta_marg, w188)
for fld in ["sum_w", "mean_w", "ess", "p_fail_weighted", "precision_w",
            "score_w", "recall_w", "cov_w", "rankIC_w_vs_success"]:
    assert chk[fld] == w188ref[fld], (fld, chk[fld], w188ref[fld])

floor_hits = {n: sum(1 for e in eps if e["slip_m"] < SLIP_FLOOR)
              for n, eps in [("calib", cal_eps), ("held", tst_eps), ("archive", arc_eps)]}

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "P50_frozen_director": P50_FROZEN,
    "P50_train_recomputed": round(p50_train, 6),
    "variant": "T189 = T188 ablation damping OFF, w=T173/(slip+eps*), eps*=0.01",
    "fric_term": "none",
    "no_damp": "ON (floor removed; raw slip)",
    "slip_floor": None,
    "slip_floor_hits_T188ref": floor_hits,
    "eps_star": EPS_STAR,
    "eps_star_rule": "max(median_calib(slip>0)=0.008358,0.01) train-locked pre-reg (best from 188)",
    "sweep_best_logonly": "eps=0.05 lift +0.19 vs T187 observed in r188, NOT refit",
    "shrink_lambda": "1.0->0 arm = pure T173 hard tag (see held_T173/trainfold_T173)",
    "w_cap_unnormalized_T189": round(1.0 / EPS_STAR, 4),
    "w_cap_unnormalized_T188ref": round(1.0 / (SLIP_FLOOR + EPS_STAR), 4),
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "trainfold_T173": hard_rule(cal_eps, lambda e: t173(e, theta_marg)),
    "held_T173": hard_rule(tst_eps, lambda e: t173(e, theta_marg)),
    "trainfold_T188ref": soft_eval(cal_eps, theta_marg, w188),
    "held_T188ref": soft_eval(tst_eps, theta_marg, w188),
    "trainfold_T189": soft_eval(cal_eps, theta_marg, w189),
    "held_T189": soft_eval(tst_eps, theta_marg, w189),
}
h3 = out["held_T173"]
h8 = out["held_T188ref"]
h9 = out["held_T189"]
t3 = out["trainfold_T173"]
t8 = out["trainfold_T188ref"]
t9 = out["trainfold_T189"]
out["lift_T189_vs_T173_held_pts"] = round(100 * (h3["p_fail_given_adm"] - h9["p_fail_weighted"]), 2)
out["lift_T189_vs_T188_held_pts"] = round(100 * (h8["p_fail_weighted"] - h9["p_fail_weighted"]), 2)
out["lift_T189_vs_T173_train_pts"] = round(100 * (t3["p_fail_given_adm"] - t9["p_fail_weighted"]), 2)
out["lift_T189_vs_T188_train_pts"] = round(100 * (t8["p_fail_weighted"] - t9["p_fail_weighted"]), 2)
# Turnover: per-episode |dw| T189-vs-T188 (same T173 support by assert).
for split, eps in [("held", tst_eps), ("trainfold", cal_eps), ("archive", arc_eps)]:
    dws = [abs(w189(e, theta_marg) - w188(e, theta_marg)) for e in eps]
    key = "turnover_T189_vs_T188_" + split
    out[key] = {"mean_abs_dw": round(sum(dws) / len(dws), 4),
                "max_abs_dw": round(max(dws), 4),
                "n_changed": sum(1 for d in dws if d > 0),
                "n_total": len(dws)}
# Breadth delta + P50 hit-rate delta.
out["breadth_delta_held"] = {"ess_T189": h9["ess"], "ess_T188": h8["ess"],
                             "mean_w_T189": h9["mean_w"], "mean_w_T188": h8["mean_w"],
                             "pct_zero_T189": h9["pct_w_zero"], "pct_zero_T188": h8["pct_w_zero"]}
out["hitrate_delta_held"] = {"score_w_T189": h9["score_w"], "score_w_T188": h8["score_w"],
                             "recall_w_T189": h9["recall_w"], "recall_w_T188": h8["recall_w"],
                             "precision_w_T189": h9["precision_w"],
                             "precision_w_T188": h8["precision_w"]}
out["director_predicted_20_35_vs_1pt"] = False  # scale check log-only
out["director_keep_bar_10pts"] = bool(out["lift_T189_vs_T188_held_pts"] > 10.0)
out["cov_ok"] = h9["cov_w"] >= h9["cov_all"] - 0.02
arc_ws = [w189(e, theta_marg) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["archive_T173_valid_rate"] = round(sum(1 for e in arc_eps if t173(e, theta_marg)) / len(arc_eps), 4)
out["verdict_rule"] = {"runA_keep": out["runA_keep_cited"],
                       "score_w_ge_70": h9["score_w"] >= 70.0,
                       "no_cov_regression": out["cov_ok"]}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["fail_path_fires"] = bool(abs(out["lift_T189_vs_T188_held_pts"]) < 1.0)
json.dump(out, open("results/aegis_v2/I2_r189_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
