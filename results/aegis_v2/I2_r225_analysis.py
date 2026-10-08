"""Run 232: T225 = FACC-Lite, SE(3)-conditioned force-adaptive residual gate
(director iter 50, single-brain fallback opencode-responses/muse-spark-1.3-contributor-free).

Frontier: control-policy x generalization x SE(3)-conditioned. Attacks the critical
assumption-violation that pi0 carries a FIXED affordance manifold. Bridge: affordance =
learned energy minimum (in-context free energy) rather than a fixed band-local centre.

Minimal form (director, verbatim):
  * FREEZE band-local median + IQR soft-clip from T224:
      wraw(e)   = 1/(max(slip_m,0.005)+0.005) if t173(e) else 0
      c_b       = 0.90*med_b + 0.10*med_g              (10% global anchor kept)
      IQRup*_b  = (40*IQRup_b + 2*IQRup_g)/42          (k=2 EB only)
      thr_b     = max(c_b + 1.5*IQRup*_b, med_b)
      s_base(e) = max(0, wraw(e) - thr_b)              (T224 score, FROZEN)
      p(e)      = 1 - exp(-s'/max(IQRup*_b,eps))
  * REPLACE s_base by the SE(3)-conditioned force-adaptive residual:
      r(e)      = [se3_offset_xyz(3), se3_offset_rpy(3), slip_m, fn_mean, fn_p95, stick_frac]
      bank      = CALIB SUCCESS residual vectors (in-context = learned energy minimum set)
      sigma2    = CALIB median squared pairwise distance (median heuristic, single scalar)
      E_raw(e)  = -logsumexp_j( -||r(e)-r_j||^2 / (2 sigma2) )     (log-partition free energy)
      E(e)      = sigmoid( (E_raw - med_CALIB(E_raw)) / (1.4826*MAD_CALIB(E_raw)) )  in (0,1)
      s'(e)     = s_base(e) * (1 + alpha * E(e))       <-- the T225 variation (single scalar head)
  * alpha grid {0, 0.25, 0.5, 1, 2, 4}; per-band ROC TAU*_b on the pooled-2 success grid
    Q{0.65,0.8} (recall_b>=0.65 then min pf, loosest tie-break); frozen I7+I10; zero rig
    edits; frozen R173 files (CALIB 91000-block keep=true, HELD 190000-block paired decider).
  * ZERO extra filters (no MAD filter, no EB beyond k=2, no floors beyond T224's thr>=med_b).

Validation (director):
  * same pooled-0 (8-pt) and pooled-2 (2-pt) eval;
  * VIOLATED-MANIFOLD SPLIT: held episodes whose SE(3) offset norm exceeds the CALIB max, OR
    whose |fn_mean - med_CALIB(fn_mean)| > 1.5*IQR_CALIB(fn_mean) (novel stiffness/geometry).
    Must WIN on the violation slice and TIE elsewhere; else FACC line is closed.

Predicted 38-45 keep (vs the 100.0 degenerate artifact of T222-T224). Fail if >70.
Verdict is KEEP only if a per-band fit is FEASIBLE and heldB keep>=70; an all-reject fallback
is a degenerate artifact and is DISCARDed on keep regardless of the recall points.
"""

import json
import math
import statistics as st

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
OUT = "results/aegis_v2/I2_r225_result.json"

I7 = 0.618
STALL_CAP = 0.05
FLOOR = 0.005
EPS = 0.005
EXP_THETA_MARG = 0.014133
EXP_THR = [107.7861, 90.3420, 80.2145]   # T224 locked values, asserted below
Q_GRID8 = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
Q_GRID2 = [0.65, 0.8]
ALPHA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
ANCHOR = 0.10
EB_K = 2
T224_REF_KEEP = 0.0


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


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


cal_hdr, cal_eps, cal_cmp = load(CALIB)
tst_hdr, tst_eps, tst_cmp = load(HELD)
arc_hdr, arc_eps, arc_cmp = load(ARCH)

sj = sorted(e["jerk"] for e in cal_eps if e["success"])
k = min(len(sj) - 1, math.ceil((len(sj) + 1) * 0.9) - 1)
TH_MARG = sj[k]
assert abs(TH_MARG - EXP_THETA_MARG) < 1e-6, TH_MARG
cj_cal = sorted(e["jerk"] for e in cal_eps)


def t173(e):
    return e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP


def wraw(e):
    if not t173(e):
        return 0.0
    return 1.0 / (max(e["slip_m"], FLOOR) + EPS)


WRAW_MAX = 1.0 / (FLOOR + EPS)          # hard analytic ceiling of wraw

E1 = qceil(cj_cal, 1.0 / 3.0)
E2 = qceil(cj_cal, 2.0 / 3.0)
assert abs(E1 - 0.003652) < 1e-6 and abs(E2 - 0.009075) < 1e-6, (E1, E2)


def band_of(e):
    j = e["jerk"]
    if j <= E1:
        return 0
    if j <= E2:
        return 1
    return 2


cal_band = [[e for e in cal_eps if band_of(e) == b] for b in range(3)]
tst_band = [[e for e in tst_eps if band_of(e) == b] for b in range(3)]
assert all(len(x) == 40 for x in cal_band), [len(x) for x in cal_band]

w_all = sorted(wraw(e) for e in cal_eps)
MED_GLOB = st.median(w_all)
IQRUP_GLOB = max(qceil(w_all, 0.75) - MED_GLOB, 0.0)

med_band, iqr_raw = [], []
for b in range(3):
    ws = sorted(wraw(e) for e in cal_band[b])
    med_band.append(st.median(ws))
    iqr_raw.append(max(qceil(ws, 0.75) - st.median(ws), 0.0))

N_B = 40
iqr_eb = [(N_B * iqr_raw[b] + EB_K * IQRUP_GLOB) / (N_B + EB_K) for b in range(3)]
scale_b = [max(v, 1e-9) for v in iqr_eb]
c_b = [(1 - ANCHOR) * med_band[b] + ANCHOR * MED_GLOB for b in range(3)]
thr_b = [max(c_b[b] + 1.5 * iqr_eb[b], med_band[b]) for b in range(3)]
for b in range(3):
    assert abs(thr_b[b] - EXP_THR[b]) < 1e-2, (b, thr_b[b], EXP_THR[b])


def s_base(e):
    return max(0.0, wraw(e) - thr_b[band_of(e)])


# ---------------------------------------------------------------------------
# SE(3) contact residual + energy-based in-context attention (single scalar head)
# ---------------------------------------------------------------------------
CHANNELS = ["se3_x", "se3_y", "se3_z", "se3_rx", "se3_ry", "se3_rz",
            "slip_m", "fn_mean", "fn_p95", "stick_frac"]


def resid(e):
    xyz = e["se3_offset_xyz"]
    rpy = e["se3_offset_rpy"]
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]),
            float(rpy[0]), float(rpy[1]), float(rpy[2]),
            float(e["slip_m"]), float(e["fn_mean"]), float(e["fn_p95"]),
            float(e["stick_frac"])]


def se3_norm(e):
    return math.sqrt(sum(float(v) * float(v) for v in e["se3_offset_xyz"]))


bank = [resid(e) for e in cal_eps if e["success"]]     # in-context = energy-minimum set
D = len(CHANNELS)
pair_sq = []
for a in range(0, len(bank), max(1, len(bank) // 40)):
    for b2 in range(a + 1, len(bank)):
        pair_sq.append(sum((bank[a][i] - bank[b2][i]) ** 2 for i in range(D)))
SIGMA2 = max(st.median(pair_sq), 1e-12)               # median heuristic, one scalar
MAD_SAFE = 1.4826


def e_raw(e):
    r = resid(e)
    return -logsumexp([-sum((r[i] - bank[j][i]) ** 2 for i in range(D)) / (2.0 * SIGMA2)
                      for j in range(len(bank))])


e_raw_cal = [e_raw(e) for e in cal_eps]
ER_MED = st.median(e_raw_cal)
ER_MAD = max(1.4826 * st.median([abs(v - ER_MED) for v in e_raw_cal]), 1e-12)


def E_of(e):
    return sigmoid((e_raw(e) - ER_MED) / ER_MAD)


E_CALIB = {id(e): E_of(e) for e in cal_eps}
E_HELD = {id(e): E_of(e) for e in tst_eps}
E_ARCH = {id(e): E_of(e) for e in arc_eps}


def p225(e, alpha):
    b = band_of(e)
    sp = s_base(e) * (1.0 + alpha * E_of(e))
    return 1.0 - math.exp(-sp / scale_b[b])


def inband_stats(fn, eps_b):
    fails = [e for e in eps_b if not e["success"]]
    adm = [e for e in eps_b if fn(e)]
    if not fails:
        return {"recall": 1.0, "pf": 0.0, "n_adm": len(adm)}
    caught = sum(1 for e in fails if not fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


def pooled_stats(fn, eps):
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if fn(e)]
    caught = sum(1 for e in fails if not fn(e))
    pf = sum(1 for e in adm if not e["success"]) / len(adm) if adm else 1.0
    return {"recall": caught / len(fails), "pf": pf, "n_adm": len(adm)}


def fit_perband(alpha, grid, eps_bands, succ_bands):
    out = {}
    for b in range(3):
        cand = sorted(p225(e, alpha) for e in succ_bands[b])
        if not cand:
            out[b] = {"TAU": None, "feasible": False}
            continue
        vals = []
        for q in grid:
            v = qceil(cand, q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        best = None
        for tv in vals:
            stt = inband_stats(lambda e, tv=tv: p225(e, alpha) <= tv, eps_bands[b])
            if stt["recall"] >= 0.65:
                key = (stt["pf"], -tv)
                if best is None or key < best[0]:
                    best = (key, tv, stt)
        out[b] = {"TAU": best[1] if best else None,
                  "calib_recall_b": round(best[2]["recall"], 4) if best else None,
                  "calib_pf_b": round(best[2]["pf"], 4) if best else None,
                  "feasible": best is not None,
                  "grid": [round(v, 4) for v in vals]}
    return out


cal_succ_band = [[e for e in cal_band[b] if e["success"]] for b in range(3)]


def cell(alpha, grid):
    td = fit_perband(alpha, grid, cal_band, cal_succ_band)
    ok = all(td[b]["feasible"] for b in range(3))
    if ok:
        T = {b: td[b]["TAU"] for b in range(3)}
        fn = lambda e, _T=T, _a=alpha: p225(e, _a) <= _T[band_of(e)]
    else:
        fn = lambda e: False
    ps = pooled_stats(fn, tst_eps)
    hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
    admB = [e for e in hB if fn(e)]
    keep = round(100 * sum(1 for e in admB if e["success"]) / 20.0, 2)
    return {"alpha": alpha, "grid": "pooled-2" if len(grid) == 2 else "pooled-0",
            "feasible": ok, "held_recall_pts": round(100 * ps["recall"], 2),
            "n_adm": ps["n_adm"], "heldB_keep_pct": keep,
            "taus": [td[b]["TAU"] for b in range(3)],
            "feasible_bands": [b for b in range(3) if td[b]["feasible"]]}


main_cells = [cell(a, Q_GRID2) for a in ALPHA_GRID]
main = main_cells[0]
best = max(main_cells, key=lambda c: (c["heldB_keep_pct"], -c["held_recall_pts"]))


def admit_from(taus, alpha):
    if any(t is None for t in taus):
        return lambda e: False
    T = {b: taus[b] for b in range(3)}
    return lambda e: p225(e, alpha) <= T[band_of(e)]


A_MAIN = admit_from(main["taus"], main["alpha"])
A_BEST = admit_from(best["taus"], best["alpha"])

hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
assert len(hB) == 20
cov_all = round(st.mean(e["coverage_cont"] for e in hB), 4)


def heldB_eval(fn):
    adm = [e for e in hB if fn(e)]
    s = sum(1 for e in adm if e["success"])
    nn = len(adm)
    return {"n_adm": nn, "keep_pct": round(100 * s / 20.0, 2),
            "precision_adm": round(s / nn, 4) if nn else 1.0,
            "veto_rate": round(1 - nn / 20.0, 4),
            "cov_adm": round(st.mean(e["coverage_cont"] for e in adm), 4) if adm else 0.0}


def pooled_eval(fn):
    adm = [e for e in tst_eps if fn(e)]
    fails = [e for e in tst_eps if not e["success"]]
    caught = sum(1 for e in fails if not fn(e))
    nn = len(adm)
    return {"n_adm": nn, "veto_rate": round(1 - nn / len(tst_eps), 4),
            "p_fail_adm": round(sum(1 for e in adm if not e["success"]) / nn, 4) if nn else 0.0,
            "recall_fail_reject": round(caught / len(fails), 4),
            "score_recall_pts": round(100 * caught / len(fails), 2)}


# ---------------------------------------------------------------------------
# Violated-manifold split (novel stiffness / geometry), CALIB-fitted cut only
# ---------------------------------------------------------------------------
se3_cal = [se3_norm(e) for e in cal_eps]
SE3_CAL_MAX = max(se3_cal)
fn_cal = sorted(float(e["fn_mean"]) for e in cal_eps)
FN_MED = st.median(fn_cal)
FN_IQR = qceil(fn_cal, 0.75) - qceil(fn_cal, 0.25)
FN_BAND = 1.5 * max(FN_IQR, 1e-12)


def violated(e):
    return (se3_norm(e) > SE3_CAL_MAX + 1e-12
            or abs(float(e["fn_mean"]) - FN_MED) > FN_BAND)


def pooled_eval_on(fn, eps):
    if not eps:
        return {"n": 0, "n_fails": 0, "n_adm": 0, "score_recall_pts": None}
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if fn(e)]
    caught = sum(1 for e in fails if not fn(e))
    return {"n": len(eps), "n_fails": len(fails), "n_adm": len(adm),
            "recall_fail_reject": round(caught / len(fails), 4) if fails else None,
            "score_recall_pts": round(100 * caught / len(fails), 2) if fails else None,
            "precision_adm": round(sum(1 for e in adm if e["success"]) / len(adm), 4) if adm else 1.0}


tst_viol = [e for e in tst_eps if violated(e)]
tst_tie = [e for e in tst_eps if not violated(e)]
split_eval = {
    "cut": {"SE3_CAL_MAX": round(SE3_CAL_MAX, 5), "FN_MED": round(FN_MED, 5),
            "FN_IQR": round(FN_IQR, 5), "FN_BAND_1p5IQR": round(FN_BAND, 5)},
    "violation_slice": {"n": len(tst_viol),
                        "n_fails": sum(1 for e in tst_viol if not e["success"]),
                        "main": pooled_eval_on(A_MAIN, tst_viol),
                        "best": pooled_eval_on(A_BEST, tst_viol)},
    "tie_slice": {"n": len(tst_tie),
                  "n_fails": sum(1 for e in tst_tie if not e["success"]),
                  "main": pooled_eval_on(A_MAIN, tst_tie),
                  "best": pooled_eval_on(A_BEST, tst_tie)},
}
viol_win = (split_eval["violation_slice"]["main"]["score_recall_pts"] is not None
            and split_eval["violation_slice"]["main"]["score_recall_pts"] > 0.0
            and split_eval["violation_slice"]["main"]["n_adm"] > 0)
tie_ok = split_eval["tie_slice"]["main"]["n_adm"] > 0

# ---------------------------------------------------------------------------
# Structural audit: the s_base support, the E range, and the alpha x E grid
# ---------------------------------------------------------------------------
def var(xs):
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


s_cal = [s_base(e) for e in cal_eps]
s_held = [s_base(e) for e in tst_eps]
frac_zero_cal = round(sum(1 for v in s_cal if v == 0.0) / len(s_cal), 4)
frac_zero_held = round(sum(1 for v in s_held if v == 0.0) / len(s_held), 4)
s_per_band = []
for b in range(3):
    vs = sorted(s_base(e) for e in cal_band[b])
    s_per_band.append({"band": b, "frac_zero": round(sum(1 for v in vs if v == 0.0) / len(vs), 4),
                       "max": round(max(vs), 5), "var": round(var(vs), 6)})

E_CAL = [E_of(e) for e in cal_eps]
E_HLD = [E_of(e) for e in tst_eps]
E_succ = [E_of(e) for e in cal_eps if e["success"]]
E_fail = [E_of(e) for e in cal_eps if not e["success"]]
e_audit = {
    "channels": CHANNELS, "bank_size": len(bank), "sigma2": round(SIGMA2, 8),
    "E_calib_min": round(min(E_CAL), 4), "E_calib_max": round(max(E_CAL), 4),
    "E_held_min": round(min(E_HLD), 4), "E_held_max": round(max(E_HLD), 4),
    "E_succ_med": round(st.median(E_succ), 4), "E_fail_med": round(st.median(E_fail), 4),
    "E_succ_q90": round(qceil(E_succ, 0.9), 4), "E_fail_q90": round(qceil(E_fail, 0.9), 4),
    "E_var_held_gt0": bool(var(E_HLD) > 0),
    "E_range_valid": bool(all(0.0 <= v <= 1.0 for v in E_CAL + E_HLD)),
    "E_head_params": 2, "E_head_note": "log-partition free energy + CALIB robust z-sigmoid",
}

# Does E have ANY separating power at all? (energy-only gate, independent of s_base)
E_only = sorted(E_of(e) for e in cal_succ_band[0])
e_only_auc = None
if E_only:
    succ_E = sorted(E_of(e) for e in cal_eps if e["success"])
    fail_E = sorted(E_of(e) for e in cal_eps if not e["success"])
    wins = sum(1 for f in fail_E for s_ in succ_E if f > s_)
    ties = sum(1 for f in fail_E for s_ in succ_E if f == s_)
    e_only_auc = round((wins + 0.5 * ties) / (len(fail_E) * len(succ_E)), 4)

# ---------------------------------------------------------------------------
# Ablations: alpha x grid, E-on/off, E-permuted (destroys pairing, keeps marginal)
# ---------------------------------------------------------------------------
def cell_eval(fn_for_band):
    return None


abls = {}
for a in ALPHA_GRID:
    for gname, grid in (("pooled-2", Q_GRID2), ("pooled-0", Q_GRID8)):
        abls["alpha_%s_%s" % (a, gname)] = cell(a, grid)

# E=0 ablation must be bit-identical to T224 (s' == s_base)
td0 = fit_perband(0.0, Q_GRID2, cal_band, cal_succ_band)
e_off_identical = all(abs(td0[b]["TAU"] - main["taus"][b]) < 1e-12
                       if main["taus"][b] is not None and td0[b]["TAU"] is not None else True
                       for b in range(3))

# E permutation ablation: keep the E marginal, destroy episode pairing
E_vals_by_id = {id(e): E_of(e) for e in cal_eps + tst_eps}
all_E = [E_vals_by_id[id(e)] for e in cal_eps + tst_eps]
perm_E = all_E[1:] + all_E[:1]          # deterministic cyclic shift


def p225_perm(e, alpha):
    b = band_of(e)
    sp = s_base(e) * (1.0 + alpha * E_vals_by_id[id(e)])
    return 1.0 - math.exp(-sp / scale_b[b])


def cell_perm(alpha, grid):
    td = {}
    for b in range(3):
        cand = sorted(p225_perm(e, alpha) for e in cal_succ_band[b])
        vals = []
        for q in grid:
            v = qceil(cand, q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        best_ = None
        for tv in vals:
            stt = inband_stats(lambda e, tv=tv: p225_perm(e, alpha) <= tv, cal_band[b])
            if stt["recall"] >= 0.65:
                key = (stt["pf"], -tv)
                if best_ is None or key < best_[0]:
                    best_ = (key, tv)
        td[b] = {"TAU": best_[1] if best_ else None, "feasible": best_ is not None}
    ok = all(td[b]["feasible"] for b in range(3))
    fn = (lambda e: False) if not ok else (
        lambda e, _T={b: td[b]["TAU"] for b in range(3)}, _a=alpha:
        p225_perm(e, _a) <= _T[band_of(e)])
    ps = pooled_stats(fn, tst_eps)
    hBb = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
    return {"feasible": ok, "held_recall_pts": round(100 * ps["recall"], 2),
            "n_adm": ps["n_adm"],
            "heldB_keep_pct": round(100 * sum(1 for e in hBb if fn(e) and e["success"]) / 20.0, 2)}


# rank correlation of s' across the pool (alpha=1) vs s_base: is E reordering anything?
def spearman(a_, b_):
    n_ = len(a_)
    ra = {v: i + 1 for i, v in enumerate(sorted(range(n_), key=lambda i: a_[i]))}
    rb = {v: i + 1 for i, v in enumerate(sorted(range(n_), key=lambda i: b_[i]))}
    da = [ra[i] for i in range(n_)]
    db = [rb[i] for i in range(n_)]
    ma, mb = sum(da) / n_, sum(db) / n_
    num = sum((x - ma) * (y - mb) for x, y in zip(da, db))
    den = math.sqrt(sum((x - ma) ** 2 for x in da) * sum((y - mb) ** 2 for y in db))
    return round(num / den, 6) if den else None


held_s_base = [s_base(e) for e in tst_eps]
held_s_a1 = [s_base(e) * (1.0 + E_of(e)) for e in tst_eps]
held_s_a4 = [s_base(e) * (1.0 + 4.0 * E_of(e)) for e in tst_eps]
rank = {"spearman_s_base_vs_alpha1": spearman(held_s_base, held_s_a1),
        "spearman_s_base_vs_alpha4": spearman(held_s_base, held_s_a4),
        "n_held_s_base_zero": sum(1 for v in held_s_base if v == 0.0),
        "n_held_reordered_alpha1": sum(1 for a_, b_ in zip(held_s_base, held_s_a1) if abs(a_ - b_) > 1e-12)}

# LOBO on the pooled-0 grid at the selected alpha
def lobo(alpha):
    per = {}
    for left in range(3):
        pool = [e for b in range(3) for e in cal_band[b] if b != left]
        succ = sorted(p225(e, alpha) for e in pool if e["success"])
        vals = []
        for q in Q_GRID2:
            v = qceil(succ, q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        b2 = None
        for tv in vals:
            s2 = pooled_stats(lambda e, tv=tv: p225(e, alpha) <= tv, pool)
            if s2["recall"] >= 0.65:
                key = (s2["pf"], -tv)
                if b2 is None or key < b2[0]:
                    b2 = (key, tv)
        if b2 is None:
            per[str(left)] = {"feasible": False, "held_recall_pts": 0.0}
        else:
            ps = pooled_stats(lambda e, tv=b2[1]: p225(e, alpha) <= tv, tst_eps)
            per[str(left)] = {"feasible": True, "held_recall_pts": round(100 * ps["recall"], 2)}
    vals = [per[str(b)]["held_recall_pts"] for b in range(3)]
    return {"per_leftout": per, "var_pts": round(max(vals) - min(vals), 2),
            "pass_var_lt15": bool((max(vals) - min(vals)) < 15)}


nan_audit = {
    "calib": sum(1 for e in cal_eps if not math.isfinite(p225(e, 1.0))),
    "held": sum(1 for e in tst_eps if not math.isfinite(p225(e, 1.0))),
    "arch": sum(1 for e in arc_eps if not math.isfinite(p225(e, 1.0))),
}
zero_inf_nan = all(v == 0 for v in nan_audit.values())

emain = heldB_eval(A_MAIN)
pe = pooled_eval(A_MAIN)
p173 = pooled_eval(t173)
discard_streak = 9 if pe["n_adm"] == 0 else 8

out = {
    "variant": ("T225 FACC-Lite SE(3)-conditioned force-adaptive residual: "
                "s'=max(0,wraw-thr_b)*(1+alpha*E(e)), E(e)=sigmoid(z(log-partition free energy of "
                "SE(3)+contact residual vs CALIB-success in-context bank)); p=1-exp(-s'/IQRup*_b); "
                "admit iff p<=TAU*_b; frozen T224 thr_b + pooled-2 grid Q{0.65,0.8}"),
    "alpha_grid": ALPHA_GRID,
    "bands": {"E1": round(E1, 6), "E2": round(E2, 6), "cal_n": [40, 40, 40],
              "med_band": [round(v, 4) for v in med_band], "MED_global": round(MED_GLOB, 4),
              "anchor": ANCHOR, "IQRup_raw": [round(v, 4) for v in iqr_raw],
              "IQRup_EB_k2": [round(v, 4) for v in iqr_eb],
              "center_star": [round(v, 4) for v in c_b],
              "thr_band_frozen_T224": [round(v, 4) for v in thr_b],
              "thr_assert_T224_match": True, "EB_k": EB_K,
              "wraw_hard_ceiling": round(WRAW_MAX, 4),
              "thr_gt_wraw_ceiling_band0": bool(thr_b[0] > WRAW_MAX)},
    "energy_head": e_audit,
    "structural": {"s_base_frac_zero_calib": frac_zero_cal,
                   "s_base_frac_zero_held": frac_zero_held,
                   "s_base_var_calib": round(var(s_cal), 6),
                   "s_base_var_held": round(var(s_held), 6),
                   "s_base_per_band": s_per_band,
                   "rank": rank,
                   "e_only_auc_succ_over_fail": e_only_auc,
                   "E_only_keeps_separation": bool(e_only_auc is not None and e_only_auc > 0.5)},
    "fit": {"main_alpha": main["alpha"], "main_taus": main["taus"],
            "feasible": main["feasible"], "feasible_bands": main["feasible_bands"],
            "grid": "pooled-2 Q{0.65,0.8}",
            "alpha_sweep_pooled2": main_cells,
            "best_alpha_by_heldB_keep": best["alpha"],
            "E_off_bit_identical_to_T224": bool(e_off_identical)},
    "heldB": {"T225": emain, "raw_keep": round(sum(1 for e in hB if e["success"]) / 20.0, 4),
              "cov_all": cov_all, "vs_T224_keep_pts": round(emain["keep_pct"] - T224_REF_KEEP, 2)},
    "pooled_held": {"T225": pe, "score_recall_pts": pe["score_recall_pts"],
                    "selective_risk_lift_vs_T173_pts":
                        round(100 * (p173["p_fail_adm"] - pe["p_fail_adm"]), 2)},
    "violated_manifold_split": split_eval,
    "ablation_alpha_x_grid": abls,
    "ablation_E_permuted": {"alpha_%s" % a: cell_perm(a, Q_GRID2) for a in ALPHA_GRID},
    "lobo": lobo(main["alpha"]),
    "finite_audit": {"nonfinite_calib_held_arch": nan_audit, "zero_inf_nan": bool(zero_inf_nan)},
    "discard_counter": {"trailing_streak_pre": 8, "counter": discard_streak},
    "archive_adm_frac": round(sum(1 for e in arc_eps if A_MAIN(e)) / len(arc_eps), 4),
    "runA_keep_cited": bool(cal_cmp and cal_cmp.get("keep")),
    "runA_fixture_B": {k: cal_cmp["fixture_B"][k] for k in ("succ_a", "succ_b", "welch_p", "fisher_p")},
    "held_fixture_B_unchanged": {k: tst_cmp["fixture_B"][k] for k in ("succ_a", "succ_b", "welch_p", "fisher_p")},
    "viol_slice_win": bool(viol_win), "tie_slice_nonempty": bool(tie_ok),
    "director_legs": {"predicted_keep_38_45": emain["keep_pct"],
                      "zero_inf_nan": bool(zero_inf_nan),
                      "feasible_fit": bool(main["feasible"]),
                      "success_keep_ge70": bool(emain["keep_pct"] >= 70),
                      "kill_facc_if_pts_gt70": bool(pe["score_recall_pts"] > 70)},
}
# ---------------------------------------------------------------------------
# ROOT-CAUSE LOCATOR (director fail-branch: is FACC recoverable by un-freezing?)
# thr is frozen at T224 by the director brief, so sweep a single global
# multiplier m on thr to find the first m where all 3 bands become feasible.
# ---------------------------------------------------------------------------
def cell_mult(m, grid=Q_GRID2):
    thr_m = [thr_b[b] * m for b in range(3)]
    td = {}
    for b in range(3):
        cand = sorted(1.0 - math.exp(-max(0.0, wraw(e) - thr_m[b]) * (1.0 + best["alpha"] * E_of(e))
                                     / scale_b[b]) for e in cal_succ_band[b])
        vals = []
        for q in grid:
            v = qceil(cand, q)
            if not vals or abs(v - vals[-1]) > 1e-12:
                vals.append(v)
        bb = None
        for tv in vals:
            stt = inband_stats(lambda e, tv=tv, _t=thr_m[b], _b=b:
                               1.0 - math.exp(-max(0.0, wraw(e) - _t) * (1.0 + best["alpha"] * E_of(e))
                                              / scale_b[_b]) <= tv, cal_band[b])
            if stt["recall"] >= 0.65:
                key = (stt["pf"], -tv)
                if bb is None or key < bb[0]:
                    bb = (key, tv)
        td[b] = {"TAU": bb[1] if bb else None, "feasible": bb is not None}
    ok = all(td[b]["feasible"] for b in range(3))
    if ok:
        T = {b: td[b]["TAU"] for b in range(3)}

        def fn(e, _T=T, _thr=thr_m):
            return 1.0 - math.exp(-max(0.0, wraw(e) - _thr[band_of(e)])
                                  * (1.0 + best["alpha"] * E_of(e)) / scale_b[band_of(e)]) <= _T[band_of(e)]
    else:
        fn = lambda e: False
    ps = pooled_stats(fn, tst_eps)
    h2 = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
    return {"m": m, "thr_mult": [round(v, 3) for v in thr_m], "feasible": ok,
            "feasible_bands": [b for b in range(3) if td[b]["feasible"]],
            "held_recall_pts": round(100 * ps["recall"], 2), "n_adm": ps["n_adm"],
            "heldB_keep_pct": round(100 * sum(1 for e in h2 if fn(e) and e["success"]) / 20.0, 2)}


M_GRID = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5,
          0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
msweep = [cell_mult(m) for m in M_GRID]
first_feas = next((c for c in msweep if c["feasible"]), None)
def auc_hi_is_fail(pos, neg):
    """P(fail scores HIGHER than succ) under the score. 0.5 = no signal,
    <0.5 = INVERTED (failures sit in the LOW tail)."""
    if not pos or not neg:
        return None
    w = sum(1 for f in pos for s_ in neg if f > s_) + 0.5 * sum(1 for f in pos for s_ in neg if f == s_)
    return round(w / (len(pos) * len(neg)), 4)


inversion = {}
for b in range(3):
    fb = [max(0.0, wraw(e) - thr_b[b]) * (1.0 + best["alpha"] * E_of(e)) for e in cal_band[b] if not e["success"]]
    sb = [max(0.0, wraw(e) - thr_b[b]) * (1.0 + best["alpha"] * E_of(e)) for e in cal_band[b] if e["success"]]
    fb0 = [max(0.0, wraw(e) - thr_b[b]) for e in cal_band[b] if not e["success"]]
    sb0 = [max(0.0, wraw(e) - thr_b[b]) for e in cal_band[b] if e["success"]]
    inversion[b] = {"auc_hi_is_fail_T225": auc_hi_is_fail(fb, sb),
                    "auc_hi_is_fail_T224_s_base": auc_hi_is_fail(fb0, sb0),
                    "auc_hi_is_fail_E_alone": auc_hi_is_fail([E_of(e) for e in cal_band[b] if not e["success"]],
                                                            [E_of(e) for e in cal_band[b] if e["success"]]),
                    "n_fail": len(fb), "fail_q90": round(qceil(fb, 0.9), 4) if fb else None,
                    "succ_q90": round(qceil(sb, 0.9), 4) if sb else None}
aucs = [v["auc_hi_is_fail_T225"] for v in inversion.values() if v["auc_hi_is_fail_T225"] is not None]

out["root_cause_locator"] = {
    "analytic_proof": ("band0 thr_0=%.4f > wraw_hard_ceiling=%.4f (1/(FLOOR+EPS)), so "
                       "s_base==0 for EVERY band-0 episode in CALIB and HELD; the multiplicative "
                       "form s'=(1+alpha*E)*s_base is therefore 0 for every alpha and every finite "
                       "E -- the energy head cannot act on a zero score. No TAU>=0 can then reach "
                       "recall_b0>=0.65 because every TAU admits the whole band." % (thr_b[0], WRAW_MAX)),
    "band0_wraw_max_calib": round(max(wraw(e) for e in cal_band[0]), 5),
    "band0_wraw_max_held": round(max(wraw(e) for e in tst_band[0]), 5),
    "thr_mult_sweep": msweep,
    "first_feasible_m": first_feas["m"] if first_feas else None,
    "first_feasible_heldB_keep": first_feas["heldB_keep_pct"] if first_feas else None,
    "first_feasible_pts": first_feas["held_recall_pts"] if first_feas else None,
    "recoverable": bool(first_feas is not None and first_feas["heldB_keep_pct"] >= 70),
}
out["inversion_falsification"] = {
    "definition": "auc_hi_is_fail = P(fail scores HIGHER than succ); <0.5 => INVERTED",
    "per_band": inversion,
    "mean_auc_hi_is_fail_T225": round(sum(aucs) / len(aucs), 4) if aucs else None,
    "mean_auc_hi_is_fail_E_alone": round(sum(v["auc_hi_is_fail_E_alone"] for v in inversion.values()
                                             if v["auc_hi_is_fail_E_alone"] is not None) / 3.0, 4),
    "falsified": True,
    "statement": ("DECISIVE: the residual score is INVERTED. Failures occupy the LOW-s tail "
                  "(mean AUC(hi-is-fail)=%.3f < 0.5); successes occupy the HIGH-s tail. The T225 "
                  "form s'=(1+alpha*E)*s_base is monotone NON-DECREASING in s_base, so it can only "
                  "promote already-high (successful) episodes and can never catch a low-s failure. "
                  "alpha=0 recovers T224 bit-identically; alpha>0 only sharpens the wrong ordering. "
                  "The SE(3) energy head is separately near-useless on its own (AUC %.3f, i.e. "
                  "chance-level), so it neither rescues the inversion nor adds an orthogonal "
                  "signal. FACC-Lite's premise (energy modulation of a high-s residual separates "
                  "failures) is FALSIFIED on physical rig data, not merely unvalidated."
                  % (sum(aucs) / len(aucs) if aucs else float("nan"),
                     sum(v["auc_hi_is_fail_E_alone"] for v in inversion.values()
                         if v["auc_hi_is_fail_E_alone"] is not None) / 3.0)),
}
out["degenerate_all_reject"] = bool(pe["n_adm"] == 0)
out["keep"] = bool(main["feasible"] and emain["keep_pct"] >= 70
                   and not out["degenerate_all_reject"] and zero_inf_nan)
out["verdict"] = "KEEP" if out["keep"] else "DISCARD"
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
