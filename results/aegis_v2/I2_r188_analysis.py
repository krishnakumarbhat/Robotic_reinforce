"""Run 188 offline analysis: T188 = T187 T173-only + tuned eps, POST-HOC ONLY.

Director iter 13 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: none, stay on T187 fric-free damped shrinker line.
Variation vs T187: tuned eps, no-damp off (damping+floor kept):
  w188(e) = 1[T173(e)] / (max(slip_m,0.005)+eps),
  eps_tuned = max(median_calib(slip_m>0), 0.01), train-locked pre-reg,
  NO fric term, I7+I10 frozen, P50 never refit (train median re-asserted log-only).
Validation (director): post-hoc sweep eps in [0.005,0.01,0.02,0.05] vs T187
  fixed 0.02 (T187 == eps 0.02 arm, bit-asserted). Abort if sweep flat vs T187.
Pre-reg verdict: KEEP iff runA_keep (cited) AND score_w188 >= 70 AND no cov regression.
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
SLIP_FLOOR = 0.005
EPS_GRID = [0.005, 0.01, 0.02, 0.05]
EPS_FLOOR = 0.01


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def w_eps(e, theta, eps):
    return 1.0 / (max(e["slip_m"], SLIP_FLOOR) + eps) if t173(e, theta) else 0.0


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

# Tuned eps: train-locked pre-reg from calib slip>0 median, floored at 0.01.
calib_pos = [e["slip_m"] for e in cal_eps if e["slip_m"] > 0]
med_pos = statistics.median(calib_pos)
EPS_TUNED = max(med_pos, EPS_FLOOR)
assert EPS_TUNED == 0.01, f"eps_tuned={EPS_TUNED} med_pos={med_pos}"
assert EPS_TUNED in EPS_GRID

# T187 identity: eps=0.02 arm must equal frozen T187 semantics (cap 40.0).
assert abs(1.0 / (SLIP_FLOOR + 0.02) - 40.0) < 1e-9
w187ref = json.load(open("results/aegis_v2/I2_r187_result.json"))["held_T187"]
chk = soft_eval(tst_eps, theta_marg, lambda e, th: w_eps(e, th, 0.02))
for fld in ["sum_w", "mean_w", "ess", "p_fail_weighted", "precision_w",
            "score_w", "recall_w", "cov_w", "rankIC_w_vs_success"]:
    assert chk[fld] == w187ref[fld], (fld, chk[fld], w187ref[fld])

floor_hits = {n: sum(1 for e in eps if e["slip_m"] < SLIP_FLOOR)
              for n, eps in [("calib", cal_eps), ("held", tst_eps), ("archive", arc_eps)]}

out = {
    "theta_marginal_logonly": round(theta_marg, 6),
    "theta_bitidentical_172_173": abs(theta_marg - 0.014133) < 1e-6,
    "n_success_calib": len(sj),
    "P50_frozen_director": P50_FROZEN,
    "P50_train_recomputed": round(p50_train, 6),
    "variant": "T188 = T187 T173-only + tuned eps, w=T173/(max(slip,0.005)+eps)",
    "fric_term": "none",
    "no_damp": "off (damping+floor kept from T187)",
    "slip_floor": SLIP_FLOOR,
    "slip_floor_hits": floor_hits,
    "eps_rule": "eps=max(median_calib(slip>0),0.01), train-locked pre-reg",
    "med_calib_slip_pos": round(med_pos, 6),
    "eps_tuned": EPS_TUNED,
    "eps_grid": EPS_GRID,
    "w_cap_unnormalized_tuned": round(1.0 / (SLIP_FLOOR + EPS_TUNED), 4),
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "trainfold_T173": hard_rule(cal_eps, lambda e: t173(e, theta_marg)),
    "held_T173": hard_rule(tst_eps, lambda e: t173(e, theta_marg)),
    "sweep": {},
}
for eps in EPS_GRID:
    out["sweep"][str(eps)] = {
        "train": soft_eval(cal_eps, theta_marg, lambda e, th, v=eps: w_eps(e, th, v)),
        "held": soft_eval(tst_eps, theta_marg, lambda e, th, v=eps: w_eps(e, th, v)),
    }
h3 = out["held_T173"]
for eps in EPS_GRID:
    h = out["sweep"][str(eps)]["held"]
    t = out["sweep"][str(eps)]["train"]
    out["sweep"][str(eps)]["lift_vs_T173_held_pts"] = round(100 * (h3["p_fail_given_adm"] - h["p_fail_weighted"]), 2)
    out["sweep"][str(eps)]["lift_vs_T173_train_pts"] = round(
        100 * (out["trainfold_T173"]["p_fail_given_adm"] - t["p_fail_weighted"]), 2)
h187 = out["sweep"]["0.02"]["held"]
for eps in EPS_GRID:
    if eps == 0.02:
        continue
    h = out["sweep"][str(eps)]["held"]
    out["sweep"][str(eps)]["lift_vs_T187_held_pts"] = round(100 * (h187["p_fail_weighted"] - h["p_fail_weighted"]), 2)
out["sweep"]["0.02"]["lift_vs_T187_held_pts"] = 0.0
out["held_T188_tuned"] = out["sweep"][str(EPS_TUNED)]["held"]
out["trainfold_T188_tuned"] = out["sweep"][str(EPS_TUNED)]["train"]
out["lift_T188_vs_T173_pts"] = out["sweep"][str(EPS_TUNED)]["lift_vs_T173_held_pts"]
out["lift_T188_vs_T187_pts"] = out["sweep"][str(EPS_TUNED)]["lift_vs_T187_held_pts"]
out["sweep_flat_vs_T187"] = all(
    abs(out["sweep"][str(e)]["lift_vs_T187_held_pts"]) < 0.05 for e in EPS_GRID if e != 0.02)
h8 = out["held_T188_tuned"]
out["cov_ok"] = h8["cov_w"] >= h8["cov_all"] - 0.02
arc_ws = [w_eps(e, theta_marg, EPS_TUNED) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["archive_T173_valid_rate"] = round(sum(1 for e in arc_eps if t173(e, theta_marg)) / len(arc_eps), 4)
out["verdict_rule"] = {"runA_keep": out["runA_keep_cited"],
                       "score_w_ge_70": h8["score_w"] >= 70.0,
                       "no_cov_regression": out["cov_ok"]}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["director_abort_flat"] = bool(out["sweep_flat_vs_T187"])
json.dump(out, open("results/aegis_v2/I2_r188_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
