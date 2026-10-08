"""Run 205: T198 = shrunk-buffered per-tool union (director iter 23).

Director iter 23 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: per-tool conformal admission (T197 46.67pts current best, run 204).
Variation (pre-reg, single eval): T197 structure x shrinkage:
  Admit198(e) := w191(e) >= TAU50*_tool[t]
              OR (w191(e) >= TAU30*_tool[t] AND jerk(e) <= THETA[t]), t=tool_id(e)
  TAU50*_tool[t] = lam[t]*TAU50_global + (1-lam[t])*TAU50_tool[t]
  TAU30*_tool[t] = lam[t]*TAU30_global + (1-lam[t])*TAU30_tool[t]
  lam[t] = 20/(n_t+20), n_t = N_calib_adm_tool[t] (train-locked counts)
  buffer_hit := jerk(e) <= THETA[t] (same as T195-T197 buffer arm)
  w191 = min(T173/(max(slip,0.005)+0.005), C99), C99=100.0 degenerate,
  fric-free, T173-only, damping ON, frozen I7+I10.
  Train-locked (recomputed + asserted bit-identical before eval):
  TAU50_global/TAU30_global (medians/Q30 over calib fitted-B admitted),
  TAU50_tool/TAU30_tool (per-tool, bit-identical to frozen T197/T196),
  THETA[t] (bit-identical to frozen T194), theta_marg 0.014133.
Why: keeps 202x203 union gain, fixes small-n tool overfit that capped 204
  (per-tool medians within 1% yet n_t~29-40: raw per-tool quantiles overfit
  noise, not scale variance; shrinkage toward global stabilises them).
Validation: same eval as 202-204 (frozen R173 calib/test/archive, frozen seeds);
  report pts (pooled failure-reject recall x100), keep% (heldB n_adm/20),
  per-tool FPR (p_fail_adm per tool); ablate A no-shrink (=T197) vs B
  no-buffer (shrunk hard arm only).
Predicted: 53-55pts, keep 60-68%.
Kill iff: score_recall <= 46.67 (no gain vs 204) OR heldB keep < 40%
  OR loses to T197 on 2/3 axes (heldB yield, pooled recall, selective-risk lift).
Frozen I7+I10, zero rig edits, no rig run, no refit. G7-clean, seg 15."""

import json
import math
import statistics as st

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
I7 = 0.618
STALL_CAP = 0.05
FLOOR = 0.005
EPS = 0.005
EXP_TAU50 = 68.0452
EXP_TAU30 = 55.9211
EXP_THETA = {0: 0.019941719703759794, 1: 0.007694750747550191,
             2: 0.01275873381323237}
EXP_THETA_MARG = 0.014133
EXP_TAU50_TOOL = {0: 76.30232298588356, 1: 76.26729063579836,
                  2: 75.47745665081642}
EXP_TAU30_TOOL = {0: 69.78764015236834, 1: 70.68624968849112,
                  2: 65.08405922571335}
EXP_NPT = {0: 29, 1: 40, 2: 39}


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def t173(e, th=None):
    th = TH_MARG if th is None else th
    return e["jerk"] <= I7 and e["jerk"] <= th and e["stall_frac"] <= STALL_CAP


def raw191(e):
    return 1.0 / (max(e["slip_m"], FLOOR) + EPS) if t173(e) else 0.0


def q193(xs, q):
    s = sorted(xs)
    i = min(len(s) - 1, math.ceil(q * len(s)) - 1)
    return s[i]


def q194(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * len(s))))]


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)
assert cal_hdr["pose_noise_cfg"] == "0.01,2" and cal_hdr["gate_mode"] == "post-hoc"
assert tst_hdr["pose_noise_cfg"] == "0.01,2" and arc_hdr["pose_noise_cfg"] == "2.0,2"

sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * 0.9) - 1)
TH_MARG = sj[k]
assert abs(TH_MARG - EXP_THETA_MARG) < 1e-6, TH_MARG

cal_adm_raws = sorted(raw191(e) for e in cal_eps if t173(e))
C99 = q193(cal_adm_raws, 0.99)
assert abs(C99 - 100.0) < 1e-9, C99


def w191(e):
    r = raw191(e)
    return min(r, C99) if r > 0 else 0.0


THETA, NB = {}, {}
for t in (0, 1, 2):
    js = [e["jerk"] for e in cal_eps if e["tool_id"] == t and e["success"]]
    NB[t] = len(js)
    THETA[t] = q194(js, 0.9)
assert all(n >= 20 for n in NB.values()), NB
for t in (0, 1, 2):
    assert abs(THETA[t] - EXP_THETA[t]) < 1e-9, (t, THETA[t])

calB = [e for e in cal_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
calB_w = sorted(w191(e) for e in calB if t173(e))
TAU50 = st.median(calB_w)
TAU30 = q193(calB_w, 0.30)
assert abs(TAU50 - EXP_TAU50) < 5e-5, TAU50
assert abs(TAU30 - EXP_TAU30) < 5e-5, TAU30

TAU50_TOOL, TAU30_TOOL, NPT = {}, {}, {}
for t in (0, 1, 2):
    adm_all = sorted(w191(e) for e in cal_eps if e["tool_id"] == t and t173(e))
    NPT[t] = len(adm_all)
    assert NPT[t] == EXP_NPT[t], (t, NPT[t])
    TAU50_TOOL[t] = st.median(adm_all)
    TAU30_TOOL[t] = q193(adm_all, 0.30)
    assert abs(TAU50_TOOL[t] - EXP_TAU50_TOOL[t]) < 1e-9, (t, TAU50_TOOL[t])
    assert abs(TAU30_TOOL[t] - EXP_TAU30_TOOL[t]) < 1e-9, (t, TAU30_TOOL[t])

# --- T198 shrinkage (decider, train-locked) ---
LAM, TAU50_S, TAU30_S = {}, {}, {}
for t in (0, 1, 2):
    LAM[t] = 20.0 / (NPT[t] + 20.0)
    TAU50_S[t] = LAM[t] * TAU50 + (1 - LAM[t]) * TAU50_TOOL[t]
    TAU30_S[t] = LAM[t] * TAU30 + (1 - LAM[t]) * TAU30_TOOL[t]
    assert TAU50_S[t] >= TAU30_S[t], (t, "shrunk buffer non-vacuous")

a198 = lambda e: (w191(e) >= TAU50_S[e["tool_id"]]) or (
    w191(e) >= TAU30_S[e["tool_id"]] and e["jerk"] <= THETA[e["tool_id"]])
a_noshrink = lambda e: (w191(e) >= TAU50_TOOL[e["tool_id"]]) or (  # ablation A = T197
    w191(e) >= TAU30_TOOL[e["tool_id"]] and e["jerk"] <= THETA[e["tool_id"]])
a_nobuf = lambda e: w191(e) >= TAU50_S[e["tool_id"]]  # ablation B shrunk hard only

hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
assert len(hB) == 20
raw_keep = sum(1 for e in hB if e["success"]) / 20.0


def heldB_eval(adm_fn):
    adm = [e for e in hB if adm_fn(e)]
    s = sum(1 for e in adm if e["success"])
    n_adm = len(adm)
    return {"n_adm": n_adm,
            "yield_sel": round(s / 20.0, 4),
            "keep_pct": round(100 * n_adm / 20.0, 2),
            "precision_adm": round(s / n_adm, 4) if n_adm else 1.0,
            "score_sel_pts": round(100 * s / n_adm, 2) if n_adm else 100.0,
            "cov_adm": round(st.mean(e["coverage_cont"] for e in adm), 4) if adm else 0.0}


e198 = heldB_eval(a198)
e_noshrink = heldB_eval(a_noshrink)
e_nobuf = heldB_eval(a_nobuf)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)

buf = [e for e in hB
       if TAU30_S[e["tool_id"]] <= w191(e) < TAU50_S[e["tool_id"]]]
buf_pass = [e for e in buf if e["jerk"] <= THETA[e["tool_id"]]]


def pooled_eval(adm_fn):
    adm = [e for e in tst_eps if adm_fn(e)]
    fails = [e for e in tst_eps if not e["success"]]
    caught = sum(1 for e in fails if not adm_fn(e))
    n_adm = len(adm)
    pf = sum(1 for e in adm if not e["success"]) / n_adm if n_adm else 0.0
    return {"n_adm": n_adm, "p_fail_adm": round(pf, 4),
            "recall_fail_reject": round(caught / len(fails), 4),
            "score_recall_pts": round(100 * caught / len(fails), 2)}


p_t173 = pooled_eval(t173)
p_198 = pooled_eval(a198)
p_noshrink = pooled_eval(a_noshrink)
p_nobuf = pooled_eval(a_nobuf)

per_tool = {}
for t in (0, 1, 2):
    rows = [e for e in tst_eps if e["tool_id"] == t]
    a8 = [e for e in rows if a198(e)]

    def pf(rs):
        return sum(1 for e in rs if not e["success"]) / len(rs) if rs else 0.0

    succ = [e for e in rows if e["success"]]
    rec8 = sum(1 for e in succ if a198(e)) / len(succ) if succ else 1.0
    per_tool[t] = {"n": len(rows),
                   "lam": round(LAM[t], 4), "n_calib_adm": NPT[t],
                   "TAU50_shrunk": round(TAU50_S[t], 4),
                   "TAU30_shrunk": round(TAU30_S[t], 4),
                   "TAU50_raw": round(TAU50_TOOL[t], 4),
                   "TAU30_raw": round(TAU30_TOOL[t], 4),
                   "THETA_tool": THETA[t],
                   "T198": {"n_adm": len(a8), "p_fail_adm_FPR": round(pf(a8), 4),
                            "success_recall": round(rec8, 4)}}

# head-to-head vs T197 (run 204) on 3 axes
axes = {"heldB_yield": e198["yield_sel"] - e_noshrink["yield_sel"],
        "pooled_recall": p_198["recall_fail_reject"] - p_noshrink["recall_fail_reject"],
        "selective_risk_lift": (p_t173["p_fail_adm"] - p_198["p_fail_adm"])
        - (p_t173["p_fail_adm"] - p_noshrink["p_fail_adm"])}
wins = sum(1 for v in axes.values() if v > 1e-12)
losses = sum(1 for v in axes.values() if v < -1e-12)

out = {
    "variant": "T198 = shrunk-buffered per-tool union: admit iff w191>=TAU50*_tool[t] OR (w191>=TAU30*_tool[t] AND jerk<=THETA[t])",
    "frozen": {"theta_marg_bitident": True, "C99": round(C99, 4),
               "C99_degenerate": True,
               "TAU50_global": round(TAU50, 4), "TAU30_global": round(TAU30, 4),
               "lam_tool": {str(k): round(v, 4) for k, v in LAM.items()},
               "TAU50_shrunk_tool": {str(k): round(v, 4) for k, v in TAU50_S.items()},
               "TAU30_shrunk_tool": {str(k): round(v, 4) for k, v in TAU30_S.items()},
               "TAU50_raw_bitident_T197": {str(k): round(v, 4) for k, v in TAU50_TOOL.items()},
               "TAU30_raw_bitident_T196": {str(k): round(v, 4) for k, v in TAU30_TOOL.items()},
               "THETA_tool_bitident": {str(k): vv for kk, vv in THETA.items() for k, vv in [(kk, vv)]},
               "N_calib_succ_tool": NB, "N_calib_adm_tool": NPT,
               "calB_adm_n": len(calB_w),
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T198": e198, "ablation_no_shrink_T197": e_noshrink,
              "ablation_no_buffer_shrunk_hard": e_nobuf, "cov_all": cov_all,
              "buffer_zone_n": len(buf), "buffer_zone_pass_jerk_n": len(buf_pass),
              "buffer_zone_all_success": bool(all(e["success"] for e in buf)) if buf else None,
              "buffer_add_vs_nobuf_yield_pts":
                  round(100 * (e198["yield_sel"] - e_nobuf["yield_sel"]), 2),
              "shrink_vs_noshrink_yield_pts":
                  round(100 * (e198["yield_sel"] - e_noshrink["yield_sel"]), 2)},
    "pooled_held": {"T173": p_t173, "T198": p_198,
                    "ablation_no_shrink_T197": p_noshrink,
                    "ablation_no_buffer": p_nobuf,
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - p_198["p_fail_adm"]), 2)},
    "per_tool_held_pooled": per_tool,
    "vs_T197_axes": {k: round(v, 4) for k, v in axes.items()},
    "vs_T197_wins": wins, "vs_T197_losses": losses,
    "kill_le_204": bool(p_198["score_recall_pts"] <= 46.67),
    "kill_keep_lt_40": bool(e198["keep_pct"] < 40.0),
    "kill_loses_2of3_vs_T197": bool(losses >= 2),
    "archive_adm_frac_T198": round(sum(1 for e in arc_eps if a198(e)) / len(arc_eps), 4),
    "cov_ok": e198["cov_adm"] >= cov_all - 0.02,
    "precision_guard_heldB": e198["precision_adm"] >= 1.0,
    "director_predicted_score_53_55": p_198["score_recall_pts"],
    "director_predicted_keep_60_68": e198["keep_pct"],
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "score_recall_ge_70": p_198["score_recall_pts"] >= 70.0,
    "precision_guard": out["precision_guard_heldB"],
    "no_cov_regression": out["cov_ok"],
    "no_kill": not (out["kill_le_204"] or out["kill_keep_lt_40"]
                    or out["kill_loses_2of3_vs_T197"])}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r198_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
