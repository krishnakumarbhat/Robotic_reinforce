"""Run 220: T213 = hierarchical global-anchored shrunk soft-score, signed, pooled-1 (director iter 38).

Director iter 38 (single-brain fallback muse-spark-1.3-contributor-free).
Frontier: LEAVE band-local pooled-0; OPEN hierarchical global-anchored frontier.
Variation:
  s = (w191 - [l*med_b + (1-l)*med_glob]) / (1.4826*max(MAD_b, MAD_glob/sqrt(n_b), 0.5*MAD_glob))
  l = n_b/(n_b+10). Signed score (no max(0,.) ReLU), no jerk term, no exp(-jerk).
Delta vs T210-T212: shrink CENTER not just denominator; drop ReLU (hides overshoot);
  drop exp(-jerk) (destroys mid-jerk separability); floor 0.5*MAD_glob (0.2 too weak).
Fit (train-locked, CALIB pooled only, single held eval, pooled-1 only):
  bands = CALIB jerk tertiles; per-band med_b/MAD_b + global med/MAD; single global
  TAU* over CALIB pooled fail-s Q{0.5..0.9} (recall>=0.65 then min pf, loosest tie-break).
Validation: pooled-1 global + leave-one-band-out (refit TAU on 2/3 bands, test left-out);
  require win on both vs T207 60.0 / plateau 46.67. Ablation: pooled-0 per-band refit of same s.
Predicted 74 keep-score, KEEP, ends 3-run discard streak. Risk: n_b large -> band-local;
  fallback T214 tune k=5/20 only.
Frozen I7+I10. Zero rig edits, frozen R173 files. G7-clean, seg 15."""

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
EXP_THETA_MARG = 0.014133
Q_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
MAD_SCALE = 1.4826
MAD_EPS = 1e-9
K_SHRINK = 10.0  # l = n_b/(n_b+k)
PLATEAU = 46.67
T207_REF = 60.0


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp_ = next((r for r in rows if r.get("record") == "compare"), None)
    return hdr, eps, cmp_


def qceil(xs, q):
    s = sorted(xs)
    i = min(len(s) - 1, math.ceil(q * len(s)) - 1)
    return s[i]


def median(xs):
    return st.median(xs)


def mad(xs, med):
    return st.median([abs(x - med) for x in xs])


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)

sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * 0.9) - 1)
TH_MARG = sj[k]
assert abs(TH_MARG - EXP_THETA_MARG) < 1e-6, TH_MARG
cal_adm_raws = sorted(
    1.0 / (max(e["slip_m"], FLOOR) + EPS)
    for e in cal_eps
    if e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP)
C99 = qceil(cal_adm_raws, 0.99)
assert abs(C99 - 100.0) < 1e-9, C99
cj_cal = sorted(e["jerk"] for e in cal_eps)


def t173(e):
    return e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP


def w191(e):
    if not t173(e):
        return 0.0
    return min(1.0 / (max(e["slip_m"], FLOOR) + EPS), C99)


E1 = qceil(cj_cal, 1.0 / 3.0)
E2 = qceil(cj_cal, 2.0 / 3.0)
assert E1 < E2


def band_of(e):
    j = e["jerk"]
    if j <= E1:
        return 0
    if j <= E2:
        return 1
    return 2


cal_band = [[e for e in cal_eps if band_of(e) == b] for b in range(3)]
assert all(len(x) == 40 for x in cal_band), [len(x) for x in cal_band]
cal_fails = [e for e in cal_eps if not e["success"]]

med_band, mad_band, n_band = [], [], []
for b in range(3):
    ws = [w191(e) for e in cal_band[b]]
    m = median(ws)
    med_band.append(m)
    mad_band.append(mad(ws, m))
    n_band.append(len(cal_band[b]))

w_all = [w191(e) for e in cal_eps]
MED_GLOB = median(w_all)
MAD_GLOB = mad(w_all, MED_GLOB)

lam_b = [n / (n + K_SHRINK) for n in n_band]
center_b = [lam_b[b] * med_band[b] + (1 - lam_b[b]) * MED_GLOB for b in range(3)]
scale_b = []
floor_leg = []
for b in range(3):
    cand = [mad_band[b], MAD_GLOB / math.sqrt(n_band[b]), 0.5 * MAD_GLOB]
    sc = MAD_SCALE * max(cand) + MAD_EPS
    if not (sc > 1e-12):
        sc = 1.0
    scale_b.append(sc)
    floor_leg.append(max(cand) > mad_band[b] + 1e-12)
floor_frac = sum(floor_leg) / 3.0


def s_hier(e):
    b = band_of(e)
    return (w191(e) - center_b[b]) / scale_b[b]


def pooled_stats(admit_fn, eps):
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if admit_fn(e)]
    caught = sum(1 for e in fails if not admit_fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


def fit_global(score_fn, eps_pool, fails_pool):
    sc_fail = sorted(score_fn(e) for e in fails_pool)
    vals = []
    for q in Q_GRID:
        v = qceil(sc_fail, q)
        if not vals or abs(v - vals[-1]) > 1e-12:
            vals.append(v)
    best = None
    for tv in vals:
        s = pooled_stats(lambda e, tv=tv: score_fn(e) >= tv, eps_pool)
        if s["recall"] >= 0.65:
            key = (s["pf"], tv)
            if best is None or key < best[0]:
                best = (key, tv, s)
    if best is None:
        return {"TAU": None, "feasible": False}
    return {"TAU": best[1], "calib_recall": round(best[2]["recall"], 4),
            "calib_pf": round(best[2]["pf"], 4), "feasible": True}


fit = fit_global(s_hier, cal_eps, cal_fails)
feas = fit["feasible"]
TAU = fit["TAU"] if feas else None
A = (lambda e: s_hier(e) >= TAU) if feas else (lambda e: False)

# pooled-0 ablation: per-band ROC refit of same hierarchical score
def fit_perband(score_fn):
    out = {}
    for b in range(3):
        fb = [e for e in cal_band[b] if not e["success"]]
        sc = sorted(score_fn(e) for e in fb)
        vals = []
        for q in Q_GRID:
            v = qceil(sc, q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        best = None
        for tv in vals:
            hb = cal_band[b]
            fails = [e for e in hb if not e["success"]]
            adm = [e for e in hb if score_fn(e) >= tv]
            rec = sum(1 for e in fails if not (score_fn(e) >= tv)) / len(fails)
            pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
            if rec >= 0.65 and (best is None or (pf, tv) < best[0]):
                best = ((pf, tv), tv, rec, pf)
        out[b] = {"TAU": best[1] if best else None, "feasible": best is not None,
                  "rec": round(best[2], 4) if best else None, "pf": round(best[3], 4) if best else None}
    return out


tau0 = fit_perband(s_hier)
feas0 = all(tau0[b]["feasible"] for b in range(3))
if feas0:
    _t0 = {b: tau0[b]["TAU"] for b in range(3)}
    A0 = lambda e, _t0=_t0: s_hier(e) >= _t0[band_of(e)]
else:
    A0 = lambda e: False

# leave-one-band-out: refit global TAU on CALIB pooled minus band b, test left-out band
lobo = {}
for b in range(3):
    pool = [e for e in cal_eps if band_of(e) != b]
    fails = [e for e in pool if not e["success"]]
    f = fit_global(s_hier, pool, fails)
    left_cal = cal_band[b]
    left_cal_f = [e for e in left_cal if not e["success"]]
    left_held = [e for e in tst_eps if band_of(e) == b]
    left_held_f = [e for e in left_held if not e["success"]]
    if f["feasible"]:
        t = f["TAU"]
        rc = sum(1 for e in left_cal_f if not (s_hier(e) >= t)) / len(left_cal_f) if left_cal_f else 1.0
        rh = sum(1 for e in left_held_f if not (s_hier(e) >= t)) / len(left_held_f) if left_held_f else 1.0
    else:
        rc, rh = None, None
    lobo[b] = {"TAU_minus_b": f["TAU"], "feasible": f["feasible"],
               "leftout_cal_recall": round(rc, 4) if rc is not None else None,
               "leftout_held_recall": round(rh, 4) if rh is not None else None,
               "n_cal": len(left_cal), "n_cal_fails": len(left_cal_f),
               "n_held": len(left_held), "n_held_fails": len(left_held_f)}
lobo_recalls = [lobo[b]["leftout_held_recall"] for b in range(3) if lobo[b]["leftout_held_recall"] is not None]
lobo_mean = round(sum(lobo_recalls) / len(lobo_recalls), 4) if lobo_recalls else None
taus = [lobo[b]["TAU_minus_b"] for b in range(3) if lobo[b]["TAU_minus_b"] is not None]
tau_range = round(max(taus) - min(taus), 4) if len(taus) == 3 else None

a205 = lambda e: w191(e) >= 74.9317  # noqa: E731
a199 = lambda e: w191(e) >= EXP_TAU50  # noqa: E731

hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
assert len(hB) == 20
raw_keep = sum(1 for e in hB if e["success"]) / 20.0


def heldB_eval(fn):
    adm = [e for e in hB if fn(e)]
    s = sum(1 for e in adm if e["success"])
    nn = len(adm)
    return {"n_adm": nn, "yield_sel": round(s / 20.0, 4),
            "keep_pct": round(100 * s / 20.0, 2),
            "precision_adm": round(s / nn, 4) if nn else 1.0,
            "veto_rate": round(1 - nn / 20.0, 4),
            "cov_adm": round(st.mean(e["coverage_cont"] for e in adm), 4) if adm else 0.0}


def pooled_eval(fn):
    adm = [e for e in tst_eps if fn(e)]
    fails_h = [e for e in tst_eps if not e["success"]]
    caught = sum(1 for e in fails_h if not fn(e))
    nn = len(adm)
    pf = sum(1 for e in adm if not e["success"]) / nn if nn else 0.0
    return {"n_adm": nn, "veto_rate": round(1 - nn / len(tst_eps), 4),
            "p_fail_adm": round(pf, 4),
            "recall_fail_reject": round(caught / len(fails_h), 4),
            "score_recall_pts": round(100 * caught / len(fails_h), 2)}


e213 = heldB_eval(A)
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)
p_t173 = pooled_eval(t173)
pe = {"T199": pooled_eval(a199), "T205": pooled_eval(a205),
      "T207_local_ref": {"n_adm": 43, "score_recall_pts": 60.0},
      "T213_pooled1": pooled_eval(A),
      "T213_pooled0_abl": pooled_eval(A0)}
pts = pe["T213_pooled1"]["score_recall_pts"]
pts0 = pe["T213_pooled0_abl"]["score_recall_pts"]
kp = e213["keep_pct"]

band_held = {}
for b in range(3):
    hb = [e for e in tst_eps if band_of(e) == b]
    hf = [e for e in hb if not e["success"]]
    if not hf or not feas:
        band_held[b] = {"n": len(hb), "n_fails": len(hf)}
        continue
    caught = sum(1 for e in hf if not (s_hier(e) >= TAU))
    band_held[b] = {"n": len(hb), "n_fails": len(hf), "recall_band": round(caught / len(hf), 4)}

win_pooled = bool(feas and pts > T207_REF)
win_lobo = bool(lobo_mean is not None and lobo_mean >= 0.65)
win_both = bool(win_pooled and win_lobo)

out = {
    "variant": "T213 = hierarchical global-anchored signed soft-score pooled-1: s=(w191-[l*med_b+(1-l)*med_glob])/(1.4826*max(MAD_b,MAD_glob/sqrt(n_b),0.5*MAD_glob)), l=n_b/(n_b+10); no ReLU, no jerk term",
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6),
              "cal_n": [len(x) for x in cal_band],
              "cal_fails": [len(x) for x in [ [e for e in cal_band[b] if not e["success"]] for b in range(3)]],
              "med_band": [round(v, 4) for v in med_band],
              "MAD_band": [round(v, 4) for v in mad_band],
              "MED_global": round(MED_GLOB, 4),
              "MAD_global": round(MAD_GLOB, 4),
              "n_band": n_band,
              "lambda_shrink": [round(v, 4) for v in lam_b],
              "center_shrunk": [round(v, 4) for v in center_b],
              "scale_hier": [round(v, 4) for v in scale_b],
              "floor_binds_scale": floor_leg, "floor_frac": round(floor_frac, 4)},
    "fit": {"K_SHRINK": K_SHRINK, "pooled": "pooled-1 global TAU",
            "TAU_STAR": round(TAU, 4) if TAU is not None else None,
            "calib_recall": fit.get("calib_recall"), "calib_pf": fit.get("calib_pf"),
            "feasible": feas,
            "abl_pooled0_perband": {str(b): tau0[b] for b in range(3)},
            "abl_pooled0_feasible": feas0},
    "perband_ROC_held_globalTAU": band_held,
    "lobo": {**{str(b): lobo[b] for b in range(3)},
             "TAU_range": tau_range, "mean_leftout_held_recall": lobo_mean},
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "heldB_raw_keep": round(raw_keep, 4),
    "heldB": {"T213": e213, "ref_T205": heldB_eval(a205), "cov_all": cov_all},
    "pooled_held": {**pe,
                    "recall_vs_T207_pts": round(pts - T207_REF, 2) if feas else None,
                    "recall_vs_plateau_pts": round(pts - PLATEAU, 2) if feas else None,
                    "pooled1_vs_pooled0_pts": round(pts - pts0, 2) if feas else None,
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p_t173["p_fail_adm"] - pe["T213_pooled1"]["p_fail_adm"]), 2)},
    "validation": {"win_pooled_vs_T207": win_pooled, "win_lobo_ge65": win_lobo,
                   "win_both": win_both, "no_jerk_term": True},
    "archive_adm_frac_T213": round(sum(1 for e in arc_eps if A(e)) / len(arc_eps), 4),
    "cov_ok": e213["cov_adm"] >= cov_all - 0.02,
    "director_legs": {"pts_ge_70": bool(feas and pts >= 70), "heldB_keep_ge_70": bool(kp >= 70),
                      "predicted_pts_74": pts, "predicted_keep": kp},
}
out["verdict"] = "KEEP" if (feas and win_both and pts >= 70 and kp >= 70 and out["cov_ok"]) else "DISCARD"
out["keep"] = bool(out["verdict"] == "KEEP" and out["runA_keep_cited"])
json.dump(out, open("results/aegis_v2/I2_r213_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
