"""Run 191 offline analysis: T191 = T188 base + joint eps*+cap, POST-HOC ONLY.

Director iter 16 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: tail-controlled T173 weighting.
Variation (pre-reg decider, single eval): w191 = min(T173/(max(slip,0.005)+0.005), C),
  C = train-p99 of raw eps=0.005 weights over calib ADMITTED eps; damping ON;
  fric-free; T173-only; frozen I7+I10.
Validation (log-only): grid eps in [0.002,0.005,0.008,0.01] x C-quantile in
  [p95,p97.5,p99,p99.5] (quantiles per-eps over calib admitted, train-locked);
  damping-OFF ablation w=min(T173/(slip+0.005),C99).
Pre-reg: KEEP iff runA_keep AND score_w191>=70 AND no cov regression.
Predicted: score 2.0pts keep>=75, no regression on T188 base.
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
EPS_DEC = 0.005
GRID_EPS = [0.002, 0.005, 0.008, 0.01]
GRID_Q = [0.95, 0.975, 0.99, 0.995]


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def raw(e, theta, eps, floor=SLIP_FLOOR):
    return 1.0 / (max(e["slip_m"], floor) + eps) if t173(e, theta) else 0.0


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


def hard_rule(eps_, admit_fn):
    suc = [e for e in eps_ if e["success"]]
    adm = [e for e in eps_ if admit_fn(e)]
    adm_s = sum(1 for e in adm if e["success"])
    rec = adm_s / len(suc) if suc else 1.0
    prec = adm_s / len(adm) if adm else 1.0
    p_fail = (len(adm) - adm_s) / len(adm) if adm else 0.0
    cov_adm = sum(e["coverage_cont"] for e in adm) / len(adm) if adm else 0.0
    cov_all = sum(e["coverage_cont"] for e in eps_) / len(eps_)
    return {"n": len(eps_), "n_adm": len(adm),
            "recall": round(rec, 4), "score": round(100 * rec, 2),
            "precision": round(prec, 4),
            "p_fail_given_adm": round(p_fail, 4),
            "cov_adm": round(cov_adm, 4), "cov_all": round(cov_all, 4)}


def soft_eval(eps_, wfn):
    ws = [wfn(e) for e in eps_]
    assert all(w >= 0.0 for w in ws)
    sw = sum(ws)
    sw2 = sum(w * w for w in ws)
    fail_w = sum(w for w, e in zip(ws, eps_) if not e["success"])
    succ_w = sw - fail_w
    n_succ = sum(1 for e in eps_ if e["success"])
    p_w = fail_w / sw if sw else 0.0
    prec_w = succ_w / sw if sw else 1.0
    rec_w = succ_w / n_succ if n_succ else 1.0
    cov_w = sum(w * e["coverage_cont"] for w, e in zip(ws, eps_)) / sw if sw else 0.0
    cov_all = sum(e["coverage_cont"] for e in eps_) / len(eps_)
    ess = sw * sw / sw2 if sw2 else 0.0
    ic_succ = spearman(ws, [1.0 if e["success"] else 0.0 for e in eps_])
    ic_cov = spearman(ws, [e["coverage_cont"] for e in eps_])
    return {"n": len(eps_), "sum_w": round(sw, 4),
            "mean_w": round(sw / len(eps_), 4),
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


def quantile(xs, q):
    s = sorted(xs)
    i = min(len(s) - 1, math.ceil(q * len(s)) - 1)
    return s[i]


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

# Train-locked caps per eps: quantile of raw over calib ADMITTED eps.
caps = {}
for eps in GRID_EPS:
    raws = sorted(raw(e, THETA, eps) for e in cal_eps if t173(e, THETA))
    assert len(raws) == 108, (eps, len(raws))
    caps[eps] = {q: quantile(raws, q) for q in GRID_Q}

C99 = caps[EPS_DEC][0.99]


def w191(e):
    r = raw(e, THETA, EPS_DEC)
    return min(r, C99) if r > 0 else 0.0


def w188f(e):
    return raw(e, THETA, 0.01)


def woff(e):  # damping-OFF ablation at decider eps/cap
    r = 1.0 / (e["slip_m"] + EPS_DEC) if t173(e, THETA) else 0.0
    return min(r, C99) if r > 0 else 0.0


# Bit-match frozen T188 held arm.
r188 = json.load(open("results/aegis_v2/I2_r188_result.json"))["held_T188_tuned"]
chk8 = soft_eval(tst_eps, w188f)
for fld in ["sum_w", "mean_w", "ess", "p_fail_weighted", "precision_w",
            "score_w", "recall_w", "cov_w", "rankIC_w_vs_success"]:
    assert chk8[fld] == r188[fld], (fld, chk8[fld], r188[fld])

h191 = soft_eval(tst_eps, w191)
t191 = soft_eval(cal_eps, w191)
h173 = hard_rule(tst_eps, lambda e: t173(e, THETA))
h188 = soft_eval(tst_eps, w188f)
hoff = soft_eval(tst_eps, woff)

out = {
    "theta_marginal_logonly": round(THETA, 6),
    "theta_bitidentical_172_173": True,
    "P50_frozen_director": P50_FROZEN,
    "P50_train_recomputed": round(p50_train, 6),
    "variant": "T191 = T188 base + joint eps*+cap, w=min(T173/(max(slip,0.005)+0.005),C99)",
    "fric_term": "none",
    "damping": "ON (decider; OFF as log-only ablation)",
    "t173_only": True,
    "slip_floor": SLIP_FLOOR,
    "eps_star": EPS_DEC,
    "eps_star_rule": "director iter-16 joint-tune decider (0.005); grid log-only",
    "caps_train_adm": {str(eps): {str(q): round(caps[eps][q], 4) for q in GRID_Q} for eps in GRID_EPS},
    "w_cap_C99_decider": round(C99, 4),
    "w_cap_raw_max_decider": round(1.0 / (SLIP_FLOOR + EPS_DEC), 4),
    "i7_binds_heldout": sum(1 for e in tst_eps if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "held_T173": h173,
    "held_T191": {kk: vv for kk, vv in h191.items() if not kk.startswith("_")},
    "trainfold_T191": {kk: vv for kk, vv in t191.items() if not kk.startswith("_")},
    "held_T188": {kk: vv for kk, vv in h188.items() if not kk.startswith("_")},
    "held_dampOFF_ablation": {kk: vv for kk, vv in hoff.items() if not kk.startswith("_")},
}
h3 = out["held_T173"]
for tag, hh in [("T188", out["held_T188"]), ("T191", out["held_T191"]),
                ("dampOFF", out["held_dampOFF_ablation"])]:
    out[f"lift_{tag}_vs_T173_held_pts"] = round(100 * (h3["p_fail_given_adm"] - hh["p_fail_weighted"]), 2)
out["lift_T191_vs_T188_held_pts"] = round(100 * (out["held_T188"]["p_fail_weighted"] - out["held_T191"]["p_fail_weighted"]), 2)
out["lift_dampOFF_vs_T191_held_pts"] = round(100 * (out["held_T191"]["p_fail_weighted"] - out["held_dampOFF_ablation"]["p_fail_weighted"]), 2)
t3 = hard_rule(cal_eps, lambda e: t173(e, THETA))
out["lift_T191_vs_T173_train_pts"] = round(100 * (t3["p_fail_given_adm"] - out["trainfold_T191"]["p_fail_weighted"]), 2)

# Log-only grid: held p_fail/score per (eps, C-quantile).
grid = {}
for eps in GRID_EPS:
    for q in GRID_Q:
        c = caps[eps][q]
        ev = soft_eval(tst_eps, lambda e, eps=eps, c=c: (min(raw(e, THETA, eps), c) if raw(e, THETA, eps) > 0 else 0.0))
        grid[f"eps{eps}_C{q}"] = {"p_fail_weighted": ev["p_fail_weighted"],
                                  "score_w": ev["score_w"], "ess": ev["ess"],
                                  "lift_vs_T173_pts": round(100 * (h3["p_fail_given_adm"] - ev["p_fail_weighted"]), 2)}
out["grid_logonly_held"] = grid

# Turnover + cap hit-rate for decider vs T188/T189-frozen + dampOFF.
r189 = json.load(open("results/aegis_v2/I2_r189_result.json"))["held_T189"]
w189chk = soft_eval(tst_eps, lambda e: (1.0 / (e["slip_m"] + 0.01) if t173(e, THETA) else 0.0))
for fld in ["p_fail_weighted", "score_w", "ess"]:
    assert w189chk[fld] == r189[fld], (fld, w189chk[fld], r189[fld])
for split, eps_ in [("held", tst_eps), ("trainfold", cal_eps), ("archive", arc_eps)]:
    for ref, fn in [("T188", w188f), ("dampOFF", woff)]:
        dws = [abs(w191(e) - fn(e)) for e in eps_]
        out[f"turnover_T191_vs_{ref}_{split}"] = {
            "mean_abs_dw": round(sum(dws) / len(dws), 4),
            "max_abs_dw": round(max(dws), 4),
            "n_changed": sum(1 for d in dws if d > 1e-12),
            "n_total": len(dws)}
    rawd = [raw(e, THETA, EPS_DEC) for e in eps_]
    capd = [w191(e) for e in eps_]
    out[f"caphit_{split}"] = {
        "frac_capped": round(sum(1 for a, b in zip(rawd, capd) if a > b + 1e-12) / len(eps_), 4),
        "n_capped": sum(1 for a, b in zip(rawd, capd) if a > b + 1e-12),
        "n_total": len(eps_)}
out["director_predicted_score_ge_75"] = out["held_T191"]["score_w"]
out["director_predicted_lift_20pts"] = out["lift_T191_vs_T173_held_pts"]
out["cov_ok"] = out["held_T191"]["cov_w"] >= out["held_T191"]["cov_all"] - 0.02
arc_ws = [w191(e) for e in arc_eps]
out["archive_valid_mass_rate"] = round(sum(arc_ws) / len(arc_eps), 4)
out["verdict_rule"] = {"runA_keep": out["runA_keep_cited"],
                       "score_w_ge_70": out["held_T191"]["score_w"] >= 70.0,
                       "no_cov_regression": out["cov_ok"]}
out["keep"] = bool(all(out["verdict_rule"].values()))
json.dump(out, open("results/aegis_v2/I2_r191_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
