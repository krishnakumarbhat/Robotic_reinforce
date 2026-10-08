"""Run 190 offline analysis: T190 = T188 base + winsorized tail cap, POST-HOC ONLY.

Director iter 15 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: slip-penalized sizing, still open.
Variation vs T188: T188 base (damping ON: max(slip,0.005); fric-free;
  T173-only; REJECT T189 OFF), eps*=max(median_calib(slip>0),0.01)=0.01
  train-locked pre-reg; ADD winsorize raw w at train p99 + hard cap max|w|.
Validation (director): same frozen R173 split; vs T187/T188/T189 on
  keep (score_w>=70 + runA cited + no cov regression) / pts (lift vs T173)
  / turnover (mean|max|dw|) + cap hit-rate (fraction capped).
Pre-reg: KEEP iff runA_keep AND score_w190>=70 AND no cov regression.
Predicted keep 72-75, +1.2-1.5pts; if fail, ablate cap vs eps* next (not here).
Frozen I7+I10, zero rig edits, no rig run, no refit.
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


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def raw188(e, theta):
    return 1.0 / (max(e["slip_m"], SLIP_FLOOR) + EPS_STAR) if t173(e, theta) else 0.0


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


def soft_eval(eps, wfn):
    ws = [wfn(e) for e in eps]
    assert all(w >= 0.0 for w in ws)
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
            "rankIC_w_vs_coverage": round(ic_cov, 4),
            "_ws": ws}


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
THETA = theta_marg
succ_slip = sorted(e["slip_m"] for e in cal_eps if e["success"])
p50_train = statistics.median(succ_slip)
assert abs(P50_FROZEN - p50_train) < 2e-4
calib_pos = [e["slip_m"] for e in cal_eps if e["slip_m"] > 0]
med_pos = statistics.median(calib_pos)
EPS_STAR = max(med_pos, 0.01)
assert EPS_STAR == 0.01, f"eps*={EPS_STAR} med_pos={med_pos}"

# Train-locked winsor cap: p99 of RAW T188 weights over calib ADMITTED (T173) eps.
raw_cal_adm = sorted(raw188(e, THETA) for e in cal_eps if t173(e, THETA))
assert len(raw_cal_adm) == 108, len(raw_cal_adm)
qi = min(len(raw_cal_adm) - 1, math.ceil(0.99 * len(raw_cal_adm)) - 1)
W_CAP = raw_cal_adm[qi]
assert W_CAP > 0 and W_CAP <= 1.0 / (SLIP_FLOOR + EPS_STAR) + 1e-9


def w190(e):
    r = raw188(e, THETA)
    return min(r, W_CAP) if r > 0 else 0.0


def w188f(e):
    return raw188(e, THETA)


def w189f(e):
    return 1.0 / (e["slip_m"] + EPS_STAR) if t173(e, THETA) else 0.0


def w187f(e):
    return 1.0 / (max(e["slip_m"], SLIP_FLOOR) + 0.02) if t173(e, THETA) else 0.0


# Bit-match frozen T188/T189 held arms from result files.
r188 = json.load(open("results/aegis_v2/I2_r188_result.json"))["held_T188_tuned"]
r189 = json.load(open("results/aegis_v2/I2_r189_result.json"))["held_T189"]
chk8 = soft_eval(tst_eps, w188f)
chk9 = soft_eval(tst_eps, w189f)
for fld in ["sum_w", "mean_w", "ess", "p_fail_weighted", "precision_w",
            "score_w", "recall_w", "cov_w", "rankIC_w_vs_success"]:
    assert chk8[fld] == r188[fld], (fld, chk8[fld], r188[fld])
    assert chk9[fld] == r189[fld], (fld, chk9[fld], r189[fld])

floor_hits = {n: sum(1 for e in eps if e["slip_m"] < SLIP_FLOOR)
              for n, eps in [("calib", cal_eps), ("held", tst_eps), ("archive", arc_eps)]}

h190 = soft_eval(tst_eps, w190)
t190 = soft_eval(cal_eps, w190)
h173 = hard_rule(tst_eps, lambda e: t173(e, THETA))
h187 = soft_eval(tst_eps, w187f)
h188 = soft_eval(tst_eps, w188f)
h189 = soft_eval(tst_eps, w189f)

out = {
    "theta_marginal_logonly": round(THETA, 6),
    "theta_bitidentical_172_173": True,
    "P50_frozen_director": P50_FROZEN,
    "P50_train_recomputed": round(p50_train, 6),
    "variant": "T190 = T188 base + winsorized tail cap, w=min(T173/(max(slip,0.005)+eps*),p99)",
    "fric_term": "none",
    "damping": "ON (T188 base; T189 OFF rejected)",
    "t173_only": True,
    "slip_floor": SLIP_FLOOR,
    "slip_floor_hits": floor_hits,
    "eps_star": EPS_STAR,
    "eps_star_rule": "max(median_calib(slip>0),0.01) train-locked pre-reg",
    "med_calib_slip_pos": round(med_pos, 6),
    "w_cap_p99_train_adm": round(W_CAP, 4),
    "w_cap_unnormalized_raw": round(1.0 / (SLIP_FLOOR + EPS_STAR), 4),
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "i7_binds_calib": sum(1 for e in cal_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "held_T173": h173,
    "held_T190": {kk: vv for kk, vv in h190.items() if not kk.startswith("_")},
    "trainfold_T190": {kk: vv for kk, vv in t190.items() if not kk.startswith("_")},
    "held_T187": {kk: vv for kk, vv in h187.items() if not kk.startswith("_")},
    "held_T188": {kk: vv for kk, vv in h188.items() if not kk.startswith("_")},
    "held_T189": {kk: vv for kk, vv in h189.items() if not kk.startswith("_")},
}
h3 = out["held_T173"]
for tag, hh in [("T187", out["held_T187"]), ("T188", out["held_T188"]),
                ("T189", out["held_T189"]), ("T190", out["held_T190"])]:
    out[f"lift_{tag}_vs_T173_held_pts"] = round(100 * (h3["p_fail_given_adm"] - hh["p_fail_weighted"]), 2)
out["lift_T190_vs_T187_held_pts"] = round(100 * (out["held_T187"]["p_fail_weighted"] - out["held_T190"]["p_fail_weighted"]), 2)
out["lift_T190_vs_T188_held_pts"] = round(100 * (out["held_T188"]["p_fail_weighted"] - out["held_T190"]["p_fail_weighted"]), 2)
out["lift_T190_vs_T189_held_pts"] = round(100 * (out["held_T189"]["p_fail_weighted"] - out["held_T190"]["p_fail_weighted"]), 2)
t3 = hard_rule(cal_eps, lambda e: t173(e, THETA))
out["lift_T190_vs_T173_train_pts"] = round(100 * (t3["p_fail_given_adm"] - out["trainfold_T190"]["p_fail_weighted"]), 2)
# Turnover + cap hit-rate per split vs T187/T188/T189.
for split, eps in [("held", tst_eps), ("trainfold", cal_eps), ("archive", arc_eps)]:
    for ref, fn in [("T187", w187f), ("T188", w188f), ("T189", w189f)]:
        dws = [abs(w190(e) - fn(e)) for e in eps]
        out[f"turnover_T190_vs_{ref}_{split}"] = {
            "mean_abs_dw": round(sum(dws) / len(dws), 4),
            "max_abs_dw": round(max(dws), 4),
            "n_changed": sum(1 for d in dws if d > 1e-12),
            "n_total": len(dws)}
    raw = [w188f(e) for e in eps]
    capd = [w190(e) for e in eps]
    out[f"caphit_{split}"] = {
        "frac_capped": round(sum(1 for a, b in zip(raw, capd) if a > b + 1e-12) / len(eps), 4),
        "n_capped": sum(1 for a, b in zip(raw, capd) if a > b + 1e-12),
        "n_total": len(eps)}
out["director_predicted_keep_72_75"] = out["held_T190"]["score_w"]
out["director_predicted_lift_12_15pts"] = out["lift_T190_vs_T173_held_pts"]
out["cov_ok"] = out["held_T190"]["cov_w"] >= out["held_T190"]["cov_all"] - 0.02
arc_ws = [w190(e) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["verdict_rule"] = {"runA_keep": out["runA_keep_cited"],
                       "score_w_ge_70": out["held_T190"]["score_w"] >= 70.0,
                       "no_cov_regression": out["cov_ok"]}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["fail_next"] = "ablate cap vs eps* next"
json.dump(out, open("results/aegis_v2/I2_r190_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
