"""Run 195: T195 = T191 joint weight IN-SELECTION + frozen I7+I10 (director iter 20).

Director iter 20 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: conditional-buffer admission (hybridize 193+194).
Variation vs 192/193/194 (pre-reg, single eval): keep TAU50 hard-admit, reuse
  194 THETA[tool] ONLY in buffer, not as POST-HOC gate:
  Admit195(e) := w191(e) >= TAU50
              OR (w191(e) >= TAU30 AND jerk(e) <= THETA[tool_id(e)])
  w191 = min(T173/(max(slip,0.005)+0.005), C99), C99=100.0 degenerate,
  fric-free, T173-only, damping ON, frozen I7+I10.
  Train-locked (recomputed + asserted bit-identical before eval):
  TAU50=68.0452 (median calib fitted-B admitted), TAU30=55.9211 (p30),
  THETA={t0:0.01994172,t1:0.00769475,t2:0.01275873} (90pct calib-SUCCESS jerk
  per tool), theta_marg=0.014133, I7 ceiling 0.618, stall cap 0.05.
Validation: calib fitted-B TAUs frozen check; THETA[t] from 194 recomputed;
  report keep (gated yield heldB), selective risk P(fail|admit) pooled held,
  per-tool violation (p_fail_adm per tool T195 vs T194 baseline).
Predicted: score 71.2pts (pooled failure-recall units, T194=33.3 baseline),
  keep>=70 pass via +recall in buffer with precision guard (heldB prec 1.0).
Kill iff: buffer adds <1.5pts vs 192 [100*(yield195-yield192) on heldB]
  OR any tool T195 p_fail_adm_tool > T194 baseline same tool.
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


def q193(xs, q):  # r193 quantile (ceil)
    s = sorted(xs)
    i = min(len(s) - 1, math.ceil(q * len(s)) - 1)
    return s[i]


def q194(xs, q):  # r194 pct (int)
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

# --- admission rules ---
a192 = lambda e: w191(e) >= TAU50
a193 = lambda e: w191(e) >= TAU30
a194 = lambda e: t173(e) and e["jerk"] <= THETA[e["tool_id"]]
a195 = lambda e: (w191(e) >= TAU50) or (w191(e) >= TAU30 and e["jerk"] <= THETA[e["tool_id"]])

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


e192, e193, e195 = heldB_eval(a192), heldB_eval(a193), heldB_eval(a195)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
buffer_add_pts = round(100 * (e195["yield_sel"] - e192["yield_sel"]), 2)


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
p_194 = pooled_eval(a194)
p_195 = pooled_eval(a195)

# per-tool selective risk + violation (rejection) rates, held pooled
per_tool = {}
kill_tool = []
for t in (0, 1, 2):
    rows = [e for e in tst_eps if e["tool_id"] == t]
    base = [e for e in rows if t173(e)]
    a4 = [e for e in rows if a194(e)]
    a5 = [e for e in rows if a195(e)]

    def pf(rs):
        return sum(1 for e in rs if not e["success"]) / len(rs) if rs else 0.0

    rej4 = (len(base) - len(a4)) / len(base) if base else 0.0
    rej5 = (len(base) - len(a5)) / len(base) if base else 0.0
    pf4, pf5 = pf(a4), pf(a5)
    fires = pf5 > pf(a4) + 1e-12
    kill_tool.append(fires)
    per_tool[t] = {"n": len(rows), "n_T173": len(base),
                   "T194": {"n_adm": len(a4), "p_fail_adm": round(pf4, 4),
                            "viol_reject_rate": round(rej4, 4)},
                   "T195": {"n_adm": len(a5), "p_fail_adm": round(pf5, 4),
                            "viol_reject_rate": round(rej5, 4)},
                   "kill_tool_violation_gt_194": bool(fires)}

buf_zone = [e for e in hB if TAU30 <= w191(e) < TAU50]
buf_pass = [e for e in buf_zone if e["jerk"] <= THETA[e["tool_id"]]]

out = {
    "variant": "T195 = T191 weight IN-SELECTION conditional-buffer: admit iff w>=TAU50 OR (w>=TAU30 AND jerk<=THETA[tool])",
    "frozen": {"theta_marg_bitident": True, "C99": round(C99, 4),
               "C99_degenerate": True, "TAU50": round(TAU50, 4),
               "TAU30": round(TAU30, 4), "THETA_tool": {str(k): v for k, v in THETA.items()},
               "N_calib_succ_tool": NB, "calB_adm_n": len(calB_w),
               "i7_binds_heldB": sum(1 for e in hB if e["jerk"] > I7)},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_B": [cal_cmp["fixture_B"]["succ_b"] / 20.0, cal_cmp["fixture_B"]["succ_a"] / 20.0],
    "runA_welch_p": cal_cmp["fixture_B"]["welch_p"],
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T192_TAU50": e192, "T193_TAU30": e193, "T195_hybrid": e195,
              "cov_all": cov_all,
              "buffer_zone_n": len(buf_zone),
              "buffer_zone_pass_tool_n": len(buf_pass),
              "buffer_zone_all_success": all(e["success"] for e in buf_zone)},
    "buffer_add_pts_vs_192": buffer_add_pts,
    "kill_buffer_lt_1pt5": bool(buffer_add_pts < 1.5),
    "pooled_held": {"T173": p_t173, "T194": p_194, "T195": p_195,
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - p_195["p_fail_adm"]), 2)},
    "per_tool_held_pooled": per_tool,
    "kill_any_tool_violation_gt_194": bool(any(kill_tool)),
    "archive_adm_frac_T195": round(sum(1 for e in arc_eps if a195(e)) / len(arc_eps), 4),
    "cov_ok": e195["cov_adm"] >= cov_all - 0.02,
    "precision_guard_heldB": e195["precision_adm"] >= 1.0,
    "director_predicted_score_71pt2": p_195["score_recall_pts"],
    "director_predicted_keep_ge_70": e195["keep_pct"],
}
out["verdict_rule"] = {
    "runA_keep": out["runA_keep_cited"],
    "score_recall_ge_70": p_195["score_recall_pts"] >= 70.0,
    "precision_guard": out["precision_guard_heldB"],
    "no_cov_regression": out["cov_ok"],
    "no_kill": not (out["kill_buffer_lt_1pt5"] or out["kill_any_tool_violation_gt_194"])}
out["keep"] = bool(all(out["verdict_rule"].values()))
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open("results/aegis_v2/I2_r195_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
