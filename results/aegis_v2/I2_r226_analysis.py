"""Run 233: T226 = FACC-Full, contact-conditioned ADAPTIVE threshold (director iter 51).

Director decision (verbatim intent): T225 kept a FROZEN band-local fence thr_b; T226 UNFREEZES it.
  thr_b(e) = median_band + alpha*|wrench| + SE3_bias      (clamped at a 5% lower bound)
  s(e)     = max(0, wraw(e) - thr_b(e))
  residual gate applies ONLY on contact-phase episodes, not free-space.

Why this can differ from T225 (which KILLED the FACC line): T225 modulated a score that was
IDENTICALLY ZERO (thr_0 = 107.79 > wraw ceiling 100.0), and it modulated it with a 10-D energy
head whose standalone AUC was 0.4847 = chance. T226 attacks both defects:
  (a) thr_b is rebuilt on the band MEDIAN (not med + 1.5*IQR), so thr_b < wraw ceiling and the
      score has support again -- this is the direct fix for T225 P1;
  (b) the SE(3) term is ADDITIVE and signed, and the wrench term uses the single channel that
      carries signal (peak normal force fn_p95, AUC hi-is-fail 0.582 on CALIB), not a
      scale-destroying 10-D distance.
The sign of alpha is pre-registered as SYMMETRIC {0, +-0.1 .. +-4}: with admit = {s <= tau} and
failures carrying the HIGH-fn_p95 spike, only alpha < 0 raises the fence under a failure. Both
arms are reported; the sign is a result, not a choice.

Pre-registered protocol (single held eval, no re-fit on HELD):
  wraw(e)   = 1/(max(slip_m,0.005)+0.005) if t173(e) else 0        [T173 trigger, frozen]
  t173(e)   = jerk <= 0.618 and jerk <= TH_MARG and stall_frac <= 0.05
  bands     = CALIB jerk tertiles E1=0.003652, E2=0.009075          [frozen]
  med_b     = CALIB band median of wraw                             [T221 lineage]
  scale_b   = (40*IQRup_b + 2*IQRup_g)/42                           [T224 EB, frozen]
  Z_W(e)    = (fn_p95(e) - med_CALIB)/(1.4826*MAD_CALIB)            [robust contact wrench]
  D_SE3(e)  = ||se3_offset_xyz(e)|| / max_CALIB ||.||               [pose bias, in [0,1]]
  thr(e)    = med_b + alpha*Z_W + beta*med_b*D_SE3 ; non-contact -> med_b ; clip to [5, 100]
  s(e)      = max(0, wraw - thr) ; p = 1 - exp(-s/scale_b)
  admit     <=> p <= tau_b, tau_b from CALIB success-quantile grid Q8, recall_b >= 0.65, min pf
Selection is CALIB-only (max pooled failure-recall, tie min pf, tie min |alpha|); HELD is scored
ONCE by the selected cell. The full (alpha,beta) surface is reported, not only the argmax.
Frozen I7 + I10. Zero rig edits. Frozen R173 CALIB/HELD + ARCHIVE (contact-shift split).
Verdict is KEEP only if the per-band CALIB fit is FEASIBLE, HELD fixture_B keep >= 70, and no
all-reject degeneracy. G7 additionally bans soft/adaptive gate variants from sets of KEEP, so the
G7 status is forced independent of the numbers.
"""

import json
import math
import statistics as st

CALIB = "results/aegis_v2/I2_r173_calib_0012.jsonl"
HELD = "results/aegis_v2/I2_r173_test_0012.jsonl"
ARCH = "results/aegis_v2/I2_r173_archive_20.jsonl"
OUT = "results/aegis_v2/I2_r226_result.json"

I7 = 0.618
STALL_CAP = 0.05
FLOOR = 0.005
EPS = 0.005
EXP_THETA_MARG = 0.014133
EXP_E1 = 0.003652
EXP_E2 = 0.009075
EXP_MED_B = [82.80, 75.14, 65.62]
Q_GRID8 = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
Q_GRID2 = [0.65, 0.8]
ALPHA_GRID = [0.0, -0.1, 0.1, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -2.0, 2.0, -4.0, 4.0]
BETA_GRID = [0.0, -0.5, 0.5]
EB_K = 2
CLAMP_FRAC = 0.05
WRAW_MAX = 1.0 / (FLOOR + EPS)
RECALL_MIN = 0.65
G7_ADAPTIVE_GATE_BAN = True


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    hdr = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    return hdr, eps


def qceil(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


cal_hdr, cal_eps = load(CALIB)
tst_hdr, tst_eps = load(HELD)
arc_hdr, arc_eps = load(ARCH)

sj = sorted(e["jerk"] for e in cal_eps if e["success"])
TH_MARG = sj[min(len(sj) - 1, math.ceil((len(sj) + 1) * 0.9) - 1)]
assert abs(TH_MARG - EXP_THETA_MARG) < 1e-6, TH_MARG
cj_cal = sorted(e["jerk"] for e in cal_eps)
E1 = qceil(cj_cal, 1.0 / 3.0)
E2 = qceil(cj_cal, 2.0 / 3.0)
assert abs(E1 - EXP_E1) < 1e-6 and abs(E2 - EXP_E2) < 1e-6, (E1, E2)


def t173(e):
    return e["jerk"] <= I7 and e["jerk"] <= TH_MARG and e["stall_frac"] <= STALL_CAP


def wraw(e):
    return 1.0 / (max(e["slip_m"], FLOOR) + EPS) if t173(e) else 0.0


def band_of(e):
    j = e["jerk"]
    return 0 if j <= E1 else (1 if j <= E2 else 2)


def se3_norm(e):
    return math.sqrt(sum(float(v) * float(v) for v in e["se3_offset_xyz"]))


def in_contact(e):
    return e["fn_mean"] > 0.0


cal_band = [[e for e in cal_eps if band_of(e) == b] for b in range(3)]
tst_band = [[e for e in tst_eps if band_of(e) == b] for b in range(3)]
assert all(len(x) == 40 for x in cal_band), [len(x) for x in cal_band]

med_band, iqr_raw = [], []
for b in range(3):
    ws = sorted(wraw(e) for e in cal_band[b])
    med_band.append(st.median(ws))
    iqr_raw.append(max(qceil(ws, 0.75) - st.median(ws), 0.0))
for b in range(3):
    assert abs(med_band[b] - EXP_MED_B[b]) < 0.05, (b, med_band[b], EXP_MED_B[b])

w_all = sorted(wraw(e) for e in cal_eps)
MED_GLOB = st.median(w_all)
IQRUP_GLOB = max(qceil(w_all, 0.75) - MED_GLOB, 0.0)
N_B = 40
scale_b = [max((N_B * iqr_raw[b] + EB_K * IQRUP_GLOB) / (N_B + EB_K), 1e-9) for b in range(3)]

# --- features: robust CALIB z of the contact wrench, normalised SE(3) pose bias -------------
fp_cal = [e["fn_p95"] for e in cal_eps]
FP_MED = st.median(fp_cal)
FP_MAD = max(1.4826 * st.median([abs(v - FP_MED) for v in fp_cal]), 1e-12)
FP_MAX = max(fp_cal)
SE3_CAL_MAX = max(se3_norm(e) for e in cal_eps)


def z_wrench(e):
    """Arm-1 (PRIMARY, pre-registered): robust CALIB z of the peak normal force."""
    return (e["fn_p95"] - FP_MED) / FP_MAD


def w_wrench(e):
    """Arm-2 (secondary): min-max CALIB normalisation, gain alpha is then in score units.

    Registered after arm-1 was observed to saturate the 5% clamp (FP_MAD collapses to ~6e-4
    because 75% of episodes sit in an atom at 0.4968 N while the failure spike sits at ~1.25 N,
    so the robust z reaches ~1e3 and every alpha in the grid clips to an endpoint).
    """
    return (e["fn_p95"] - FP_MED) / max(FP_MAX - FP_MED, 1e-12)


NORMS = {"robust_z": z_wrench, "minmax": w_wrench}


def d_se3(e):
    return se3_norm(e) / max(SE3_CAL_MAX, 1e-12)


CLAMP_LO = CLAMP_FRAC * WRAW_MAX


def thr_of(e, alpha, beta, clamp=True, contact_only=True, norm="robust_z"):
    b = band_of(e)
    t = med_band[b]
    if (not contact_only) or in_contact(e):
        t = t + alpha * NORMS[norm](e) + beta * med_band[b] * d_se3(e)
    if clamp:
        return min(max(t, CLAMP_LO), WRAW_MAX)
    return min(t, WRAW_MAX)


def s_of(e, alpha, beta, clamp=True, contact_only=True, norm="robust_z"):
    return max(0.0, wraw(e) - thr_of(e, alpha, beta, clamp, contact_only, norm))


def p_of(e, alpha, beta, clamp=True, contact_only=True, norm="robust_z"):
    return 1.0 - math.exp(-s_of(e, alpha, beta, clamp, contact_only, norm) / scale_b[band_of(e)])



# --------------------------------------------------------------------------- statistics ----
def auc_hi_is_fail(fail_vals, succ_vals):
    if not fail_vals or not succ_vals:
        return None
    w = sum(1 for f in fail_vals for s in succ_vals if f > s)
    w += 0.5 * sum(1 for f in fail_vals for s in succ_vals if f == s)
    return round(w / (len(fail_vals) * len(succ_vals)), 4)


def recall_pf(fn, eps):
    fails = [e for e in eps if not e["success"]]
    adm = [e for e in eps if fn(e)]
    caught = sum(1 for e in fails if not fn(e))
    pf = (sum(1 for e in adm if not e["success"]) / len(adm)) if adm else 1.0
    return (caught / len(fails) if fails else 1.0), pf, len(adm)


def fit_band(alpha, beta, eps_b, succ_b, grid, clamp, contact_only, norm):
    cand = sorted(p_of(e, alpha, beta, clamp, contact_only, norm) for e in succ_b)
    if not cand:
        return {"tau": None, "feasible": False}
    vals = []
    for q in grid:
        v = qceil(cand, q)
        if not vals or abs(v - vals[-1]) > 1e-12:
            vals.append(v)
    best = None
    for tv in vals:
        r, pf, n_adm = recall_pf(
            lambda e, t=tv: p_of(e, alpha, beta, clamp, contact_only, norm) <= t, eps_b)
        if r >= RECALL_MIN:
            key = (pf, -tv)
            if best is None or key < best[0]:
                best = (key, tv, r, pf, n_adm)
    return {"tau": best[1] if best else None, "feasible": best is not None,
            "recall": round(best[2], 4) if best else None,
            "pf": round(best[3], 4) if best else None,
            "n_adm": best[4] if best else 0}


cal_succ_band = [[e for e in cal_band[b] if e["success"]] for b in range(3)]


def recall_ceiling(alpha, beta, norm, clamp=True, contact_only=True):
    """Analytic upper bound on failure-recall per band: s == 0 implies p == 0 implies ADMIT
    for every tau >= 0, so only failures with a strictly positive score can ever be rejected."""
    return [round(sum(1 for e in cal_band[b] if not e["success"]
                      and s_of(e, alpha, beta, clamp, contact_only, norm) > 0.0)
                / max(sum(1 for e in cal_band[b] if not e["success"]), 1), 4)
            for b in range(3)]


def cell(alpha, beta, grid=Q_GRID8, clamp=True, contact_only=True, norm="robust_z"):
    td = [fit_band(alpha, beta, cal_band[b], cal_succ_band[b], grid, clamp, contact_only, norm)
          for b in range(3)]
    ok = all(t["feasible"] for t in td)
    if ok:
        T = {b: td[b]["tau"] for b in range(3)}
        fn = lambda e: p_of(e, alpha, beta, clamp, contact_only, norm) <= T[band_of(e)]
    else:
        fn = lambda e: False
    r, pf, n_adm = recall_pf(fn, cal_eps)
    hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
    admB = [e for e in hB if fn(e)]
    thr_cal = [thr_of(e, alpha, beta, clamp, contact_only, norm) for e in cal_eps]
    return {"alpha": alpha, "beta": beta, "norm": norm,
            "grid": "Q8" if len(grid) == 8 else "Q2",
            "clamp": clamp, "contact_only": contact_only, "feasible": ok,
            "calib_recall_pts": round(100 * r, 2), "calib_pf": round(pf, 4),
            "calib_n_adm": n_adm, "feasible_bands": [b for b in range(3) if td[b]["feasible"]],
            "taus": [td[b]["tau"] for b in range(3)],
            "recall_ceiling_band": recall_ceiling(alpha, beta, norm, clamp, contact_only),
            "clamp_bind_frac": round(
                sum(1 for e, t in zip(cal_eps, thr_cal)
                    if t <= CLAMP_LO + 1e-9 or t >= WRAW_MAX - 1e-9) / len(cal_eps), 4),
            "heldB_keep_pct": round(100 * sum(1 for e in admB if e["success"]) / 20.0, 2),
            "heldB_n_adm": len(admB)}


SURFACE = [cell(a, b_, norm=n) for n in ("robust_z", "minmax")
           for a in ALPHA_GRID for b_ in BETA_GRID]
FEAS = [c for c in SURFACE if c["feasible"]]
SEL = None
if FEAS:
    SEL = min(FEAS, key=lambda c: (-c["calib_recall_pts"], c["calib_pf"],
                                   abs(c["alpha"]), abs(c["beta"])))
# reference cell for the root-cause analysis when nothing is feasible: the cell whose
# analytic recall ceiling is largest in the WORST band (deterministic, no held peeking).
REF = min(SURFACE, key=lambda c: (-min(c["recall_ceiling_band"]), abs(c["alpha"]), abs(c["beta"])))


def admit_from(c):
    if c is None or any(t is None for t in c["taus"]):
        return lambda e: False
    T = {b: c["taus"][b] for b in range(3)}
    a, b_, cl, co, nm = c["alpha"], c["beta"], c["clamp"], c["contact_only"], c["norm"]
    return lambda e: p_of(e, a, b_, cl, co, nm) <= T[band_of(e)]



A_SEL = admit_from(SEL)

# ------------------------------------------------------------------- HELD paired decider ---
hB = [e for e in tst_eps if e["suite"] == "fixture_B" and e["path_mode"] == "fitted"]
assert len(hB) == 20, len(hB)
COV_CHAMP = round(st.mean(e["coverage_cont"] for e in hB), 4)
admB = [e for e in hB if A_SEL(e)]
keep_pct = round(100 * sum(1 for e in admB if e["success"]) / 20.0, 2)
discard_pct = round(100 * (1 - len(admB) / 20.0), 2)


def contact_slice_eval(fn, eps, label):
    con = [e for e in eps if in_contact(e)]
    free = [e for e in eps if not in_contact(e)]
    out = {"slice": label, "n": len(eps), "n_contact": len(con), "n_free_space": len(free)}
    for nm, sub in (("contact", con), ("free_space", free)):
        a = [e for e in sub if fn(e)]
        out[nm] = {
            "n": len(sub), "n_adm": len(a),
            "keep_pct": round(100 * sum(1 for e in a if e["success"]) / len(sub), 2) if sub else None,
            "discard_pct": round(100 * (1 - len(a) / len(sub)), 2) if sub else None,
            "contact_success_pct": round(100 * sum(1 for e in a if e["success"]) / len(a), 2) if a else None,
            "cov_adm": round(st.mean([e["coverage_cont"] for e in a]), 4) if a else 0.0,
        }
    return out


HELD_SLICES = contact_slice_eval(A_SEL, tst_eps, "held_0.01,2")
ARCH_SLICES = contact_slice_eval(A_SEL, arc_eps, "archive_2.0,2_contact_shift")
n_free_total = sum(1 for e in tst_eps + arc_eps if not in_contact(e))

# --------------------------------------------------------------------------- diagnostics ----
# The root-cause analysis always runs, on the selected cell if one is feasible, otherwise on the
# deterministic REF cell (max analytic recall ceiling in the worst band).
WORK = SEL or REF
a, b_, nm = WORK["alpha"], WORK["beta"], WORK["norm"]
thr_cal = [thr_of(e, a, b_, norm=nm) for e in cal_eps]
s_cal = [s_of(e, a, b_, norm=nm) for e in cal_eps]
p_cal = [p_of(e, a, b_, norm=nm) for e in cal_eps]
per_band = {}
for bi in range(3):
    fb = [s_of(e, a, b_, norm=nm) for e in cal_band[bi] if not e["success"]]
    sb = [s_of(e, a, b_, norm=nm) for e in cal_band[bi] if e["success"]]
    per_band[bi] = {
        "auc_hi_is_fail_s_T226": auc_hi_is_fail(fb, sb),
        "auc_hi_is_fail_wrench_alone": auc_hi_is_fail(
            [NORMS[nm](e) for e in cal_band[bi] if not e["success"]],
            [NORMS[nm](e) for e in cal_band[bi] if e["success"]]),
        "auc_hi_is_fail_se3_alone": auc_hi_is_fail([d_se3(e) for e in cal_band[bi] if not e["success"]],
                                                   [d_se3(e) for e in cal_band[bi] if e["success"]]),
        "n_fail": len(fb), "fail_s_q50": round(st.median(fb), 4) if fb else None,
        "succ_s_q50": round(st.median(sb), 4) if sb else None,
    }

# PROPOSITION 1 -- analytic recall ceiling.  s == 0 => p == 0 => ADMIT for every tau >= 0, so a
# failure can only ever be rejected when wraw(e) > thr_b(e).  This bound is threshold-free.
CEIL_A0 = recall_ceiling(0.0, 0.0, "robust_z")
BEST_CEIL = min(SURFACE, key=lambda c: (-min(c["recall_ceiling_band"]), abs(c["alpha"])))
CEIL_BEST = BEST_CEIL["recall_ceiling_band"]

# PROPOSITION 2 -- the alpha grid is a no-op: the 5% clamp saturates at an endpoint for every
# non-zero alpha, so each cell collapses onto the alpha=0 rule or onto the all-reject corner.
sat = {}
for n in ("robust_z", "minmax"):
    for aa in ALPHA_GRID:
        c = next(x for x in SURFACE if x["norm"] == n and x["alpha"] == aa and x["beta"] == 0.0)
        sat["%s:a=%s" % (n, aa)] = {"clamp_bind_frac": c["clamp_bind_frac"],
                                    "recall_ceiling_band": c["recall_ceiling_band"],
                                    "feasible": c["feasible"]}

# PROPOSITION 3 -- the wrench channel carries signal but its robust scaling is degenerate.
fp_lo = [e["fn_p95"] for e in cal_eps if e["fn_p95"] < (FP_MED + FP_MAX) / 2]
spike = [e["fn_p95"] for e in cal_eps if e["fn_p95"] >= (FP_MED + FP_MAX) / 2]
spike_fail = sum(1 for e in cal_eps if e["fn_p95"] >= (FP_MED + FP_MAX) / 2 and not e["success"])
n_fail_cal = sum(1 for e in cal_eps if not e["success"])

D = {
    "working_cell": {"role": "selected" if SEL else "ref_max_ceiling",
                     **{k: WORK[k] for k in ("alpha", "beta", "norm", "grid", "clamp", "contact_only",
                                             "calib_recall_pts", "calib_pf", "calib_n_adm",
                                             "feasible_bands", "taus", "recall_ceiling_band")}},
    "P1_support_and_threshold_free_recall_ceiling": {
        "statement": ("s == 0 implies p == 0 implies ADMIT for every tau >= 0, so failure-recall is "
                      "upper-bounded by the fraction of failures with wraw > thr_b -- independent of "
                      "the tau grid. That ceiling is below the 0.65 criterion in every band at "
                      "alpha=0, so the per-band fit cannot be feasible in the alpha=0 limit."),
        "alpha0_ceiling_band": CEIL_A0, "alpha0_ceiling_max": max(CEIL_A0),
        "recall_min_criterion": RECALL_MIN,
        "alpha0_n_fail_band": [sum(1 for e in cal_band[b] if not e["success"]) for b in range(3)],
        "alpha0_fails_reachable_band": [
            sum(1 for e in cal_band[b] if not e["success"] and wraw(e) > med_band[b]) for b in range(3)],
        "best_ceiling_cell": {"alpha": BEST_CEIL["alpha"], "beta": BEST_CEIL["beta"],
                              "norm": BEST_CEIL["norm"], "ceiling_band": CEIL_BEST,
                              "ceiling_max": max(CEIL_BEST), "feasible": BEST_CEIL["feasible"]},
        "ceiling_exceeds_criterion_anywhere": bool(max(CEIL_BEST) >= RECALL_MIN),
        "support": {"thr_min_calib": round(min(thr_cal), 4), "thr_max_calib": round(max(thr_cal), 4),
                    "wraw_hard_ceiling": round(WRAW_MAX, 4), "clamp_lo": round(CLAMP_LO, 4),
                    "thr_below_ceiling_frac": round(sum(1 for v in thr_cal if v < WRAW_MAX)
                                                     / len(thr_cal), 4),
                    "s_frac_zero": round(sum(1 for v in s_cal if v == 0.0) / len(s_cal), 4),
                    "s_var": round(st.pvariance(s_cal), 4)},
    },
    "P2_alpha_grid_is_a_no_op": {
        "statement": ("The 5% lower clamp is not a safety rail on this data, it is the whole "
                      "mechanism: for every non-zero alpha the term alpha*W saturates thr at one "
                      "endpoint (W_max ~ 1.6e3 under the robust z, alpha*W_max >> 100), so each cell "
                      "collapses onto either the alpha=0 fence or the thr == ceiling all-reject "
                      "corner. The 78-cell surface is therefore flat."),
        "cells": len(SURFACE), "cells_feasible": len(FEAS),
        "cells_clamp_binding_over_half": sum(1 for c in SURFACE if c["clamp_bind_frac"] > 0.5),
        "distinct_recall_ceilings": len({tuple(c["recall_ceiling_band"]) for c in SURFACE}),
        "per_alpha_zero_beta": sat,
    },
    "P3_wrench_signal_real_but_scaling_degenerate": {
        "fp_med": round(FP_MED, 6), "fp_mad": round(FP_MAD, 8), "fp_max": round(FP_MAX, 6),
        "fp_low_cluster_n": len(fp_lo), "fp_low_cluster_range": [round(min(fp_lo), 6), round(max(fp_lo), 6)],
        "fp_gap_over_mad": round((FP_MAX - FP_MED) / FP_MAD, 2),
        "fp_spike_n": len(spike), "fp_spike_range": [round(min(spike), 6), round(max(spike), 6)],
        "spike_failures": spike_fail, "n_fail_calib": n_fail_cal,
        "spike_fail_frac": round(spike_fail / max(n_fail_cal, 1), 4),
        "z_wrench_max": round(max(z_wrench(e) for e in cal_eps), 2),
        "minmax_wrench_max": round(max(w_wrench(e) for e in cal_eps), 4),
        "auc_hi_is_fail_fn_p95_alone": auc_hi_is_fail(
            [e["fn_p95"] for e in cal_eps if not e["success"]],
            [e["fn_p95"] for e in cal_eps if e["success"]]),
        "spike_is_bimodal_atom": bool(FP_MAD < 0.01 * (FP_MAX - FP_MED)),
        "statement": ("fn_p95 is the only channel with real failure signal (AUC %.3f; %d/%d failures "
                      "sit in the high-force spike), but 75%% of episodes sit in an atom at 0.4968 N "
                      "so the robust MAD is %.2e N against a signal gap of %.3f N. The robust z is "
                      "therefore numerically degenerate (max %.0f) and can only act as a hard switch, "
                      "never as a graded fence -- which is why the min-max arm was registered."
                      % (auc_hi_is_fail([e["fn_p95"] for e in cal_eps if not e["success"]],
                                        [e["fn_p95"] for e in cal_eps if e["success"]]) or float("nan"),
                         spike_fail, n_fail_cal, FP_MAD, FP_MAX - FP_MED,
                         max(z_wrench(e) for e in cal_eps))),
    },
    "per_band_separation": per_band,
    "channel_stats": {"FP_MED": round(FP_MED, 4), "FP_MAD": round(FP_MAD, 8), "FP_MAX": round(FP_MAX, 4),
                      "se3_cal_max": round(SE3_CAL_MAX, 6),
                      "med_band": [round(v, 4) for v in med_band],
                      "scale_band": [round(v, 4) for v in scale_b]},
}

# -------------------------------------------------------------------------- ablations -------
ABL = {
    "alpha0_beta0_frozen_median": cell(0.0, 0.0, norm="robust_z"),
    "alpha0_minmax": cell(0.0, 0.0, norm="minmax"),
    "clamp_off": cell(a, b_, clamp=False, norm=nm),
    "beta0_wrench_only": cell(a, 0.0, norm=nm),
    "se3_only": cell(0.0, b_, norm=nm),
    "contact_restriction_off": cell(a, b_, contact_only=False, norm=nm),
    "grid_Q2": cell(a, b_, grid=Q_GRID2, norm=nm),
    "alpha_sign_scan": [{"alpha": aa,
                         "feasible_robust_z": any(c["alpha"] == aa and c["norm"] == "robust_z" and c["feasible"] for c in SURFACE),
                         "feasible_minmax": any(c["alpha"] == aa and c["norm"] == "minmax" and c["feasible"] for c in SURFACE)}
                        for aa in ALPHA_GRID],
}
ABL["clamp_off_identical_to_sel"] = (
    None if SEL is None else ABL["clamp_off"]["heldB_keep_pct"] == SEL["heldB_keep_pct"])



# ------------------------------------------------------------------------------- LOBO -----
# LOBO is only meaningful when at least one band has a failure that a positive score can reach;
# otherwise every leave-out fit is provably infeasible and the spread is undefined, not zero.
CALIB_HAS_REACHABLE_FAILURE = bool(SEL is not None or max(CEIL_BEST) > 0.0)


def lobo():
    """Leave-one-band-out: refit the rule on 2 bands, score the held-out band."""
    out = []
    for b in range(3):
        other = [x for x in range(3) if x != b]
        eps_o = [e for bi in other for e in cal_band[bi]]
        succ_o = [e for bi2 in other for e in cal_succ_band[bi2]]
        if SEL is None and not CALIB_HAS_REACHABLE_FAILURE:
            out.append({"held_out_band": b, "recall": None, "feasible": False})
            continue
        med_o = st.median([wraw(e) for e in eps_o])
        iqr_o = max(qceil([wraw(e) for e in eps_o], 0.75) - med_o, 0.0)
        sc_o = (2 * N_B * iqr_o + EB_K * IQRUP_GLOB) / (2 * N_B + EB_K)

        def sc(e, med_o=med_o, sc_o=sc_o, b_=b_, a_=a, nm_=nm):
            t = med_o + a_ * NORMS[nm_](e) + b_ * med_o * d_se3(e)
            t = min(max(t, CLAMP_LO), WRAW_MAX) if in_contact(e) else med_o
            t = min(t, WRAW_MAX)
            s = max(0.0, wraw(e) - t)
            bi = band_of(e)
            bsc = scale_b[bi] if bi in other else sc_o
            return 1.0 - math.exp(-s / bsc)

        cand = sorted(sc(e) for e in succ_o)
        vals = [qceil(cand, q) for q in Q_GRID8]
        vals = sorted({round(v, 12) for v in vals})
        best = None
        for tv in vals:
            r, pf, _ = recall_pf(lambda e, t=tv: sc(e) <= t, cal_band[b])
            if r >= RECALL_MIN and (best is None or pf < best[0]):
                best = (pf, tv, r)
        out.append({"held_out_band": b, "recall": round(best[2], 4) if best else None,
                    "feasible": best is not None})
    rs = [o["recall"] for o in out if o["recall"] is not None]
    spread = round(max(rs) - min(rs), 4) if rs else None
    return {"per_band": out, "recall_spread": spread,
            "pass": bool(rs and len(rs) == 3 and spread <= 0.15),
            "feasible_bands": sum(1 for o in out if o["feasible"])}


LOBO = lobo()

# ------------------------------------------------------------------ numeric self-checks ----
vals = [p_of(e, a, b_, norm=nm) for e in tst_eps] + [p_of(e, a, b_, norm=nm) for e in arc_eps]
zero_inf_nan = sum(1 for v in vals if not math.isfinite(v))
zero_score = [s_of(e, 0.0, 0.0) for e in cal_eps]

degenerate = bool(SEL is None or recall_pf(A_SEL, tst_eps)[2] == 0)
keep_flag = bool(SEL is not None and SEL["feasible"] and keep_pct >= 70
                 and not degenerate and zero_inf_nan == 0)

out = {
    "run": 233, "segment": 15, "idea": "I2/T226",
    "rig": "frozen R173 canonical-rig episodes (PyBullet DIRECT, physics_contact points)",
    "command": "timeout 1200 python3 results/aegis_v2/I2_r226_analysis.py",
    "compare_champion": "trochoid (physical champion: B success 1.00, Fisher p=0.0083 vs raster)",
    "frozen_constants": {"I7": I7, "TH_MARG": round(TH_MARG, 6), "E1": round(E1, 6), "E2": round(E2, 6),
                         "med_band": [round(v, 4) for v in med_band], "scale_band": [round(v, 4) for v in scale_b],
                         "clamp_lo": round(CLAMP_LO, 4), "wraw_ceiling": round(WRAW_MAX, 4)},
    "rule": "thr(e)=clip(med_b + alpha*W(fn_p95) + beta*med_b*D_SE3, 5, 100); s=max(0,wraw-thr); "
            "p=1-exp(-s/scale_b); admit <=> p<=tau_b; non-contact episodes fall back to med_b",
    "alpha_grid": ALPHA_GRID, "beta_grid": BETA_GRID, "tau_grid": Q_GRID8,
    "norms": ["robust_z (primary)", "minmax (registered after arm-1 clamp saturation)"],
    "grid_cells": len(SURFACE), "grid_feasible": len(FEAS),
    "surface": SURFACE,
    "selected_cell": SEL,
    "ref_cell": REF,
    "heldB": {"keep_pct": keep_pct, "discard_pct": discard_pct, "n_adm": len(admB),
              "coverage_cont_champion_all20": COV_CHAMP,
              "coverage_cont_admitted": round(st.mean([e["coverage_cont"] for e in admB]), 4) if admB else 0.0,
              "precision_admitted": round(sum(1 for e in admB if e["success"]) / len(admB), 4) if admB else 1.0},
    "contact_slices": [HELD_SLICES, ARCH_SLICES],
    "free_space_episodes_total": n_free_total,
    "contact_restriction_active": n_free_total > 0,
    "diagnostics": D,
    "ablations": ABL,
    "lobo": LOBO,
    "zero_inf_nan": zero_inf_nan,
    "alpha0_s_finite": all(math.isfinite(v) for v in zero_score),
    "degenerate_all_reject": degenerate,
    "keep": keep_flag,
    "g7_adaptive_gate_ban_applies": G7_ADAPTIVE_GATE_BAN,
    "verdict": "KEEP" if (keep_flag and not G7_ADAPTIVE_GATE_BAN) else "DISCARD",
}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("surface", "diagnostics", "ablations")},
                 indent=1))
print("--- P1 recall ceiling per band ---")
print(json.dumps(D["P1_support_and_threshold_free_recall_ceiling"], indent=1))
print("--- P3 wrench channel ---")
print(json.dumps(D["P3_wrench_signal_real_but_scaling_degenerate"], indent=1))
print("--- surface summary ---")
for c in SURFACE:
    print("  %-9s a=%5s b=%5s feas=%-5s bands=%-12s ceil=%-22s clamp=%5s calib_pts=%6s pf=%6s heldB_keep=%6s"
          % (c["norm"], c["alpha"], c["beta"], c["feasible"], c["feasible_bands"],
             c["recall_ceiling_band"], c["clamp_bind_frac"], c["calib_recall_pts"],
             c["calib_pf"], c["heldB_keep_pct"]))
