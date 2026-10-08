"""Run 192: T192 = T191 joint weight IN-SELECTION (director iter 17).

Director iter 17 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: CLOSE post-hoc-only; OPEN in-loop-W (post-hoc cannot move keep by
  def -- explains 189/190/191 flat 1.0).
Variation (pre-reg, single eval): SAME T191 weight
  w192 = min(T173/(max(slip,0.005)+0.005), C99), C99 train-p99 over calib
  admitted (T173) raws, damping ON, fric-free, T173-only, frozen I7+I10 --
  applied PRE-KEEP/RANK as episode admission, not post-hoc reweighting:
  Admit(e) := w192(e) >= TAU, TAU = median of w192 over CALIB fitted-B
  admitted (train-locked, no tuning). Gated yield keep_sel = #(succ&admit)/20
  on HELD fitted-B (G2 population). Raw physics success values are never
  scaled (G7: admission changes the keep SET, never per-episode
  coverage/success arithmetic; cf T173 hard_rule cov_adm precedent).
  Episode-slip is known only post-episode, so pre-keep selection === offline
  admission on frozen R173 episodes: DELTA vs 189-191 is aggregation timing
  only (same eps*=0.005, floor 0.005, C99) -- isolates causality.
Validation (A/B vs T188 post-hoc on same held population): gated keep_sel vs
  raw keep; rank-shift mass = P(admit_T173 XOR admit_T192); P(w=C99) cap-hit;
  slip dist admitted vs rejected; log-only TAU grid {p25,p75} train-locked.
Pre-reg: KEEP iff runA_keep AND score_sel>=70 AND no cov regression.
Predicted: keep_sel 0.68-0.75 (~70pts vs raw 0.95/1.0); first break of invariance.
KILL: if keep_sel still ~raw (within 1 ep = 0.05), kill T188-weight family
  entirely; pivot to selection/threshold frontier.
Frozen I7+I10, zero rig edits, no rig run, no refit. G7-clean, seg 15."""

import json
import math
import statistics

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
ALPHA = 0.1
I7_THETA = 0.618
STALL_CAP = 0.05
SLIP_FLOOR = 0.005
EPS_STAR = 0.005


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, theta):
    return (e["jerk"] <= I7_THETA and e["jerk"] <= theta
            and e["stall_frac"] <= STALL_CAP)


def raw(e, theta, eps=EPS_STAR, floor=SLIP_FLOOR):
    return 1.0 / (max(e["slip_m"], floor) + eps) if t173(e, theta) else 0.0


def quantile(xs, q):
    s = sorted(xs)
    i = min(len(s) - 1, math.ceil(q * len(s)) - 1)
    return s[i]


def iqr(xs):
    return (round(quantile(xs, 0.25), 4), round(quantile(xs, 0.75), 4)) if xs else (0.0, 0.0)


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)
assert cal_hdr["pose_noise_cfg"] == "0.01,2" and cal_hdr["gate_mode"] == "post-hoc"
assert tst_hdr["pose_noise_cfg"] == "0.01,2" and arc_hdr["pose_noise_cfg"] == "2.0,2"
assert cal_hdr["path_mode"] == "fitted"

sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * (1 - ALPHA)) - 1)
THETA = sj[k]
assert abs(THETA - 0.014133) < 1e-6, f"rig drift: theta={THETA}"

# C99 train-locked: p99 of raw over ALL calib T173-admitted (== T191 construction).
cal_adm_raws = sorted(raw(e, THETA) for e in cal_eps if t173(e, THETA))
assert len(cal_adm_raws) == 108, len(cal_adm_raws)
C99 = quantile(cal_adm_raws, 0.99)
assert abs(C99 - 100.0) < 1e-9, C99  # degenerate: == raw max 1/(0.005+0.005)


def w192(e):
    r = raw(e, THETA)
    return min(r, C99) if r > 0 else 0.0


def w188(e):  # T188 post-hoc arm (A/B baseline): eps=0.01, floor 0.005, no cap
    return 1.0 / (max(e["slip_m"], SLIP_FLOOR) + 0.01) if t173(e, THETA) else 0.0


# T188 post-hoc arm recomputed directly (same formula; frozen file cross-check log-only).

# TAU train-locked: median w192 over CALIB fitted-B admitted ONLY.
calB = [e for e in cal_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
assert len(calB) == 20 and sum(1 for e in calB if e["success"]) == 20
calB_adm_w = sorted(w192(e) for e in calB if t173(e, THETA))
assert len(calB_adm_w) >= 10, len(calB_adm_w)  # train support for median
TAU50 = statistics.median(calB_adm_w)
TAU25 = quantile(calB_adm_w, 0.25)
TAU75 = quantile(calB_adm_w, 0.75)

# Decider population: HELD fitted-B (G2 metric population, n=20).
hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
assert len(hB) == 20
raw_keep = sum(1 for e in hB if e["success"]) / 20.0


def select_eval(eps_, tau):
    adm = [e for e in eps_ if w192(e) >= tau]
    rej = [e for e in eps_ if w192(e) < tau]
    n_adm = len(adm)
    s_adm = sum(1 for e in adm if e["success"])
    n = len(eps_)
    yld = s_adm / n  # selection-gated yield: P(success & admit)
    prec = s_adm / n_adm if n_adm else 1.0
    cov_adm = sum(e["coverage_cont"] for e in adm) / n_adm if n_adm else 0.0
    cov_all = sum(e["coverage_cont"] for e in eps_) / n
    return {"n": n, "n_adm": n_adm, "n_rej": len(rej),
            "yield_sel": round(yld, 4), "score_sel_pts": round(100 * prec, 2),
            "precision_adm": round(prec, 4),
            "cov_adm": round(cov_adm, 4), "cov_all": round(cov_all, 4),
            "slip_adm_med": round(statistics.median([e["slip_m"] for e in adm]), 4) if adm else 0.0,
            "slip_rej_med": round(statistics.median([e["slip_m"] for e in rej]), 4) if rej else 0.0,
            "slip_adm_iqr": iqr([e["slip_m"] for e in adm]),
            "slip_rej_iqr": iqr([e["slip_m"] for e in rej])}


dec = select_eval(hB, TAU50)
g25 = select_eval(hB, TAU25)
g75 = select_eval(hB, TAU75)

# Rank-shift mass: T192 admission vs T173 hard admission on held-B fitted.
mass = sum(1 for e in hB if t173(e, THETA) != (w192(e) >= TAU50)) / len(hB)
# (T192 adm => T173 adm always since TAU50 > 0; mass == P(T173 adm & rejected).)
assert all((w192(e) >= TAU50) <= t173(e, THETA) for e in hB)

# Cap-hit P(w=C99): raw > C99 among admitted (degenerate expectation: 0).
def caphit(eps_):
    adm = [e for e in eps_ if t173(e, THETA)]
    hit = sum(1 for e in adm if raw(e, THETA) > C99 + 1e-12)
    return {"n_adm": len(adm), "n_hit": hit,
            "p_hit": round(hit / len(adm), 4) if adm else 0.0}


# A/B vs T188 post-hoc on same held-B population (weighted fail-rate, invariant keep).
def soft_pfail(eps_, wfn):
    ws = [wfn(e) for e in eps_]
    sw = sum(ws)
    fw = sum(w for w, e in zip(ws, eps_) if not e["success"])
    return round(fw / sw, 4) if sw else 0.0


h173_adm = sum(1 for e in hB if t173(e, THETA))
h173_succ = sum(1 for e in hB if e["success"] and t173(e, THETA))

out = {
    "variant": "T192 = T191 joint weight IN-SELECTION, w=min(T173/(max(slip,0.005)+0.005),C99)",
    "theta_bitidentical_172_173": True,
    "eps_star": EPS_STAR, "slip_floor": SLIP_FLOOR,
    "C99_train_adm": round(C99, 4), "C99_raw_max": round(1.0 / (SLIP_FLOOR + EPS_STAR), 4),
    "TAU50_train_median_calB_adm": round(TAU50, 4),
    "TAU25_logonly": round(TAU25, 4), "TAU75_logonly": round(TAU75, 4),
    "calB_adm_n": len(calB_adm_w),
    "i7_binds_heldB": sum(1 for e in hB if not (e["jerk"] <= I7_THETA)),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB_T173_adm": h173_adm, "heldB_T173_adm_succ": h173_succ,
    "decider_T192": dec,
    "grid_logonly_TAU": {"p25": g25, "p50-decider": dec, "p75": g75},
    "sensitivity_yield_drop_pts": round(100 * (raw_keep - dec["yield_sel"]), 2),
    "kill_fires_keep_still_raw": bool(abs(raw_keep - dec["yield_sel"]) < 0.05),
    "rank_shift_mass_vs_T173": round(mass, 4),
    "caphit": {"calib_all": caphit(cal_eps), "held_all": caphit(tst_eps),
               "heldB_fitted": caphit(hB), "archive_all": caphit(arc_eps)},
    "AB_vs_T188_posthoc_heldB": {
        "T188_p_fail_weighted": soft_pfail(hB, w188),
        "T192_selection_note": "post-hoc reweighting leaves keep invariant by construction; "
                               "in-selection moves gated yield (see sensitivity_yield_drop_pts)"},
    "archive_adm_frac": round(sum(1 for e in arc_eps if w192(e) >= TAU50) / len(arc_eps), 4),
    "cov_ok": dec["cov_adm"] >= dec["cov_all"] - 0.02,
    "director_predicted_yield_068_075": dec["yield_sel"],
}
out["verdict_rule"] = {"runA_keep": out["runA_keep_cited"],
                       "score_sel_ge_70": dec["score_sel_pts"] >= 70.0,
                       "no_cov_regression": out["cov_ok"]}
out["keep"] = bool(all(out["verdict_rule"].values()))
json.dump(out, open("results/aegis_v2/I2_r192_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
