#!/usr/bin/env python3
"""N202 -- the loop-amplitude dose.  Closes N201's OPEN TERM (the <=0.018 tool offset in
the residual-yaw law, ordering not monotone in r_eff) with a CAUSAL dose on the one
knob that could carry it: the trochoid loop offset amplitude A = AEGIS_TROCH_AMP_M
(an existing R29 knob; the rig logs the dimensionless cusp parameter k = w*A per
episode as troch_wA_dimensionless).

TWO HARNESS FACTS this readout is built around, both found the hard way:
  H1 ARM MIXING.  A run made with --compare writes BOTH arms into ONE jsonl, and the
     episode record carries no arm label (keys: path_mode, seed, suite, ... -- no
     label/arm field), so keying episodes by seed silently keeps ONE arm and drops the
     other.  The arms are contiguous per suite (candidate block then baseline block),
     so they are split BY POSITION here.  Validated on N202_CONTROL_s20_amp0p015, the
     champion amplitude against itself: both halves are bit-identical.
  H2 THE COMPARE RECORD'S a/b NAMING.  In the rig's compare record `mean_a`/`mean_b`
     and `succ_a`/`succ_b` are (BASELINE, CANDIDATE) -- `a = by_mode[base_m]`,
     `b = by_mode[cand]` -- and because the candidate runs FIRST, mean_a is the SECOND
     block of each suite and mean_b the FIRST, which is what the H1 split returns.
     (An earlier cut of this file took H2 to mean "candidate = second block" and
     inverted the arms, flipping the sign of every dose delta; corrected here.)
  H3 A baseline arm only restores the knobs named in --compare-env.  N202d overrode
     AMP but not W, so its baseline arm ran w = 8.333 instead of the champion's
     33.333: that compare record is NOT a champion pairing and is reported as invalid.

Pre-registered predictions, all decided on physics-contact episodes only:
  P1 SUPPLY   coverage_cont falls monotonically as A falls, because A sits in the
              denominator of the supply term r_eff + A.
  P2 COLLAPSE the coverage-vs-x curve, x = rho*|residual yaw|/(r_eff + A), is
              INVARIANT under the dose (a non-geometric channel would break it).
  P3 C1 BOUND legality in A is governed by the DIMENSIONLESS k = w*A through
              max_turn_deg, and the plan-level law reproduces the rig's own
              AssertionError numbers.
Inputs: results/aegis_v2/N202*_r312_*.jsonl, N202_kinematic.json.  Output:
results/aegis_v2/N202_readout.json.  No rig byte is read for scoring; coverage_cont and
success come only from the rig's own pts_source=physics_contact records.
"""
from __future__ import annotations

import glob
import json
import math

import numpy as np
from scipy import stats

AMP_STUDY = {  # stem -> (A, w, seeds, suites, pairing_valid)
    "N202_r312_s100_amp0p005": (0.005, 33.333333333333336, 100, "ABR", True),
    "N202b_r312_s200_amp0p010": (0.010, 33.333333333333336, 200, "A", True),
    "N202c_r312_s200_amp0p0075": (0.0075, 33.333333333333336, 200, "A", True),
    "N202b_r312_s200_amp0p015": (0.015, 33.333333333333336, 200, "A", True),
    "N202_r312_s100_amp0p015": (0.015, 33.333333333333336, 100, "ABR", True),
    "N202e_r312_s100_amp0p0165": (0.0165, 33.333333333333336, 100, "ABR", True),
    "N202d_r312_s100_amp0p060_w8p33": (0.060, 8.333333333333334, 100, "ABR", False),
}
R_EFF = {0: 0.035, 1: 0.040, 2: 0.050}       # min(half[0], half[1]) of TOOL_SHAPES
C1_THRESHOLD_DEG = 60.0


def load(stem: str) -> dict:
    """Purpose: read one rig file and return its episodes split into the two arms BY
    POSITION (H1) plus the compare record.  Inputs: file stem.  Outputs:
    {"cand": {suite: {seed: rec}}, "base": {...}, "compare": rec|None, "header": rec}.
    ARMS ARE LABELLED FROM THE RIG SOURCE, not from a convention: `modes = [(args.path,
    False), (args.compare, True)]` (kaggle_aegis_sweep.py:2047) so the CANDIDATE is the
    FIRST block of every suite and the baseline the second, while in the compare record
    `a = by_mode[base_m]`, `b = by_mode[cand]` (:2095-2099) so mean_a/succ_a are the
    (second-block) BASELINE and mean_b/succ_b the (first-block) CANDIDATE.  Verified on
    N202b_r312_s200_amp0p010: first block 0.7154 == mean_b, second 0.7204 == mean_a ==
    the champion value measured in the both-arms-0.015 control file."""
    recs = [json.loads(line) for line in open(f"results/aegis_v2/{stem}.jsonl")]
    hdr = next(r for r in recs if r.get("record") == "header")
    cmp_rec = next((r for r in recs if r.get("record") == "compare"), None)
    eps = [r for r in recs if r.get("record") == "episode"
           and r.get("pts_source") == "physics_contact" and "reg_yaw_plan_deg" in r]
    out: dict = {"cand": {}, "base": {}, "compare": cmp_rec, "header": hdr}
    for suite in ("fixture_A", "fixture_B", "fixture_R"):
        sub = [r for r in eps if r["suite"] == suite]
        h = len(sub) // 2
        if not h:
            continue
        out["cand"][suite] = {r["seed"]: r for r in sub[:h]}    # FIRST block (modes[0])
        out["base"][suite] = {r["seed"]: r for r in sub[h:]}    # second block
    return out


def yaw_channel() -> tuple:
    """Purpose: the coverage-vs-|residual yaw| line on fixture_A measured on the N201
    physical episodes, used as the PREDICTION engine for P2 (what an EQUIVALENT yaw
    shift of the size the amplitude dose applies through the denominator would have
    cost).  Inputs: results/aegis_v2/N201_r311_*.jsonl.  Outputs:
    (slope covc per deg, intercept, pearson r, n)."""
    ys, cs = [], []
    for path in sorted(glob.glob("results/aegis_v2/N201_r311_*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if (r.get("record") == "episode" and r.get("pts_source") == "physics_contact"
                    and r.get("suite") == "fixture_A" and "reg_yaw_plan_deg" in r):
                ys.append(abs(r["reg_yaw_plan_deg"]))
                cs.append(r["coverage_cont"])
    ys_a, cs_a = np.array(ys), np.array(cs)
    slope, intercept = np.polyfit(ys_a, cs_a, 1)
    return (float(slope), float(intercept),
            float(np.corrcoef(ys_a, cs_a)[0, 1]), len(ys))


def patch_rho(r: dict) -> float:
    """Purpose: the scrub-patch circumradius the rig uses to place rows: rho =
    hypot(half_u, side/2) with half_u = 0.20 and side = 0.12 (elongated) / 0.18 (round).
    A residual yaw theta rotates the plan about the patch centroid, so a rim point is
    displaced by rho*theta -- the DEMAND side of the law.  Inputs: episode record.
    Outputs: rho in metres."""
    shape = (r.get("fixture_spec") or {}).get("tank_shape", "round")
    side = 0.12 if shape == "elongated" else 0.18
    return math.hypot(0.20, 0.5 * side)


def xvar(r: dict, amp: float) -> float:
    """Purpose: the collapsed variable, demand over supply:
    x = rho*|residual yaw| / (r_eff + A).  Inputs: episode record, amplitude A.
    Outputs: dimensionless x."""
    return patch_rho(r) * abs(r["reg_yaw_plan_deg"]) / (R_EFF[r["tool_id"]] + amp)


def arm_stats(d: dict) -> dict:
    """Purpose: per-suite coverage/success means for one arm.  Inputs: an arm dict
    {suite: {seed: rec}}.  Outputs: {suite: {n, coverage_cont, success, min, sem}}."""
    out = {}
    for suite, seeds in d.items():
        c = np.array([r["coverage_cont"] for r in seeds.values()])
        s = np.array([int(r["success"]) for r in seeds.values()])
        out[suite] = {"n": len(c), "coverage_cont": float(c.mean()),
                      "sem": float(c.std(ddof=1) / math.sqrt(len(c))),
                      "min_coverage_cont": float(c.min()),
                      "success": float(s.mean()), "success_n": int(s.sum())}
    return out


def paired_test(cand: dict, base: dict) -> dict:
    """Purpose: the G4 paired statistics on the SHARED seeds of the two arms (same
    friction, tool, customer, noise by construction of the rig's seed rule).
    Inputs: two arm dicts.  Outputs: per-suite delta/Welch/Fisher/paired-Wilcoxon."""
    out = {}
    for suite in cand:
        shared = sorted(set(cand[suite]) & set(base[suite]))
        if not shared:
            continue
        a = np.array([cand[suite][s]["coverage_cont"] for s in shared])
        b = np.array([base[suite][s]["coverage_cont"] for s in shared])
        sa = np.array([int(cand[suite][s]["success"]) for s in shared])
        sb = np.array([int(base[suite][s]["success"]) for s in shared])
        out[suite] = {
            "n": len(shared),
            "covc_cand": float(a.mean()), "covc_base": float(b.mean()),
            "delta_covc": float(a.mean() - b.mean()),
            "welch_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
            "paired_wilcoxon_p": (None if np.all(a == b)
                                  else float(stats.wilcoxon(a, b).pvalue)),
            "n_seeds_differing": int((a != b).sum()),
            "succ_cand_n": int(sa.sum()), "succ_base_n": int(sb.sum()),
            "fisher_p_success": float(stats.fisher_exact(
                [[int(sa.sum()), len(shared) - int(sa.sum())],
                 [int(sb.sum()), len(shared) - int(sb.sum())]])[1]),
        }
    return out


def main() -> None:
    res: dict = {"harness_facts": {
        "H1_arm_mixing": "a --compare run writes BOTH arms into one jsonl and the episode "
                         "record has no arm label, so the arms are split BY POSITION: "
                         "modes[0] = candidate FIRST, baseline SECOND (rig source "
                         "kaggle_aegis_sweep.py:2047). Verified three ways: the "
                         "champion-vs-champion control is bit-identical, the first block "
                         "equals the compare record's mean_b, and the second block equals "
                         "mean_a AND equals the champion value of the control file. "
                         "CORRECTED this iteration: an earlier cut of this readout "
                         "labelled the arms the other way round, which flipped the sign "
                         "of every P1/P2 dose delta.",
        "H2_compare_naming": "in the rig compare record mean_a/mean_b and succ_a/succ_b are "
                             "(BASELINE, CANDIDATE) -- a = by_mode[base_m], b = by_mode[cand] "
                             "(kaggle_aegis_sweep.py:2095-2099) -- so mean_a is the SECOND "
                             "block and mean_b the FIRST",
        "H3_partial_knob_restore": "a baseline arm restores only the knobs named in "
                                   "--compare-env, so N202d (AMP overridden, W not) has a "
                                   "baseline at w=8.333 and its compare record is NOT a "
                                   "champion pairing",
    }}
    arms: dict[str, dict] = {}
    for stem, (amp, w, seeds, suites, valid) in AMP_STUDY.items():
        try:
            d = load(stem)
        except FileNotFoundError:
            continue
        arms[stem] = d
        c = d["compare"] or {}
        res.setdefault("files", {})[stem] = {
            "A_m": amp, "w": w, "k_wA": amp * w, "seeds": seeds, "suites": suites,
            "pairing_valid": valid,
            "aegis_reg": d["header"].get("aegis_reg"),
            "pose_noise_cfg": d["header"].get("pose_noise_cfg"),
            "rig_version": d["header"].get("rig_version"),
            "pts_source": "physics_contact",
            "compare_keep": c.get("keep"),
            "compare_means_a_baseline_b_candidate": {
                s: {"covc_a_base": c[s]["mean_a"], "covc_b_cand": c[s]["mean_b"],
                    "delta": c[s]["delta"], "welch_p": c[s]["welch_p"],
                    "succ_a_base": c[s]["succ_a"], "succ_b_cand": c[s]["succ_b"],
                    "fisher_p": c[s]["fisher_p"]}
                for s in ("fixture_A", "fixture_B", "fixture_R") if s in c},
        }

    # ---- P1: the dose, per arm, on the pairs whose baseline IS the champion --------
    dose = {}
    for stem, (amp, w, _n, _s, valid) in AMP_STUDY.items():
        if stem not in arms or amp == 0.015:
            continue
        d = arms[stem]
        dose[f"A={amp} (w={w:.2f}, k={amp * w:.3f}) pairing_valid={valid}"] = {
            "stem": stem, "candidate": arm_stats(d["cand"]), "baseline": arm_stats(d["base"]),
            "paired": paired_test(d["cand"], d["base"]),
        }
    res["P1_amplitude_dose"] = dose

    # ---- P2: does the dose act through the SUPPLY denominator? --------------------
    # If x = rho*|yaw|/(r_eff + A) were the coverage law, lowering A would move x
    # exactly the way an EQUIVALENT yaw shift does, so the fixture_A yaw line measured
    # on the N201 episodes predicts the dose's coverage cost.  The test is run on
    # fixture_A (a ROUND face: residual yaw = the 56 deg prior in every episode, so the
    # dose moves x through the denominator ALONE) and the delta is PAIRED against the
    # file's own champion baseline arm on the same seeds.
    inv: dict = {}
    slope, intercept, yaw_r, yaw_n = yaw_channel()
    ref = arms.get("N202b_r312_s200_amp0p015", {}).get("cand", {}).get("fixture_A")
    if ref:
        x0 = np.array([xvar(r, 0.015) for r in ref.values()])
        for stem in ("N202c_r312_s200_amp0p0075", "N202b_r312_s200_amp0p010",
                     "N202_r312_s100_amp0p005", "N202e_r312_s100_amp0p0165"):
            if stem not in arms:
                continue
            amp = AMP_STUDY[stem][0]
            rows = list(arms[stem]["cand"]["fixture_A"].values())
            x = np.array([xvar(r, amp) for r in rows])
            pd_ = paired_test(arms[stem]["cand"], arms[stem]["base"])["fixture_A"]
            dyaw = np.array([
                abs(r["reg_yaw_plan_deg"]) * (R_EFF[r["tool_id"]] + 0.015)
                / (R_EFF[r["tool_id"]] + amp) - abs(r["reg_yaw_plan_deg"])
                for r in rows])
            pred = slope * float(dyaw.mean())
            delta = pd_["delta_covc"]
            inv[f"A={amp}"] = {
                "n": len(rows),
                "x_mean": float(x.mean()),
                "x_shift_pct_vs_champion": float(100 * (x.mean() - x0.mean()) / x0.mean()),
                "equivalent_yaw_shift_deg": float(dyaw.mean()),
                "covc_delta_paired_vs_champion": delta,
                "predicted_delta_from_fixture_A_yaw_line": pred,
                "yaw_line": {"slope_covc_per_deg": slope, "intercept": intercept,
                             "pearson_r": yaw_r, "n": yaw_n},
                "measured_over_predicted": (None if abs(pred) < 1e-9
                                            else float(delta / pred)),
                "reading": ("if A were part of the denominator the yaw line predicts "
                            "this delta; measured/predicted far below 1 -> A carries "
                            "only a fraction of the weight a supply term needs"),
            }
    res["P2_supply_denominator"] = inv

    # ---- P3: the C1 legality law in k = w*A --------------------------------------
    kin = json.load(open("results/aegis_v2/N202_kinematic.json"))
    rows_k = [r for r in kin["rows"] if isinstance(r["max_turn_deg"], (int, float))]
    kk = [r["k"] for r in rows_k]
    tt = [r["max_turn_deg"] for r in rows_k]
    lo, hi = 0.2, 0.8
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.interp(mid, kk, tt)) < C1_THRESHOLD_DEG:
            lo = mid
        else:
            hi = mid
    k_star = 0.5 * (lo + hi)
    champ = next(r for r in rows_k if abs(r["A_m"] - 0.015) < 1e-9)
    res["P3_c1_law"] = {
        "source": kin["law_source"] if "law_source" in kin else "N202_kinematic.py",
        "predicted_max_turn_deg": {f"k={r['k']:.4f}": round(r["max_turn_deg"], 2)
                                   for r in rows_k},
        "rig_asserted_max_turn_deg": {"k=0.6000 (A=0.018)": 61.7,
                                      "k=0.7500 (A=0.0225)": 102.9,
                                      "k=1.0000 (A=0.030, elongated)": 150.6},
        "C1_threshold_deg": C1_THRESHOLD_DEG,
        "k_star_c1_boundary": k_star,
        "A_max_at_champion_rate_m": k_star / 33.333333333333336,
        "champion_A_m": 0.015, "champion_k": 0.5,
        "champion_max_turn_deg": champ["max_turn_deg"],
        "champion_C1_budget_used_frac": champ["max_turn_deg"] / C1_THRESHOLD_DEG,
        "k_invariance_line_at_k_0p5": kin["k_0p5_invariance"],
        "k_invariance_spread_deg": kin["k_0p5_max_turn_spread_deg"],
        "cusp_law": "min|T| = 1 - k, so the cusp is k = 1 for ANY (A, w) (rig R29 note)",
        "exactness": "k alone is NOT the whole invariant: at fixed k the max turn still "
                     "spans 15.8 deg over an 8x amplitude range, because the offset is "
                     "sampled once per base point so the phase step is w*BASE_DS_M",
    }

    # ---- the invalid pairing, reported as such ------------------------------------
    res["H3_invalid_pairing"] = {
        "file": "results/aegis_v2/N202d_r312_s100_amp0p060_w8p33.jsonl",
        "why": "--compare-env overrode AEGIS_TROCH_AMP_M only; the baseline arm kept "
               "AEGIS_TROCH_W=8.333 from the environment, so it is A=0.015, w=8.333, "
               "k=0.125 -- not the champion",
        "candidate_arm_absolute": arm_stats(arms["N202d_r312_s100_amp0p060_w8p33"]["cand"])
        if "N202d_r312_s100_amp0p060_w8p33" in arms else None,
        "cross_file_paired_vs_champion_same_seeds": paired_test(
            arms["N202d_r312_s100_amp0p060_w8p33"]["cand"],
            arms["N202_r312_s100_amp0p015"]["base"]) if (
                "N202d_r312_s100_amp0p060_w8p33" in arms
                and "N202_r312_s100_amp0p015" in arms) else None,
    }

    res["episodes_total"] = sum(
        sum(len(v) for v in d["cand"].values()) + sum(len(v) for v in d["base"].values())
        for d in arms.values())
    res["new_rig_bytes"] = 0
    res["metric_quantum_B"] = {
        "grid_unit": 1.0 / 128.0,
        "observed_spacing_fixture_B": 2.0 / 128.0,
        "note": "N199 measured six even-k values over 600 B episodes, so 2/128 = 0.0156 "
                "is the usable quantum on the anchored suite and 1/128 the grid unit; "
                "fixture_A is on k/96 and fixture_R on k/192.",
    }
    # every N202 file on disk, including the champion-vs-champion control and the three
    # files that died on the rig's own C1 assertion (they carry no metric).
    all_eps = 0
    for path in glob.glob("results/aegis_v2/N202*.jsonl"):
        for line in open(path):
            try:
                if json.loads(line).get("record") == "episode":
                    all_eps += 1
            except json.JSONDecodeError:
                continue
    res["episodes_all_n202_files"] = all_eps
    json.dump(res, open("results/aegis_v2/N202_readout.json", "w"), indent=1)

    for k, v in dose.items():
        print(f"  DOSE {k}")
        for suite, p in v["paired"].items():
            print(f"    {suite}: covc {p['covc_cand']:.4f} vs {p['covc_base']:.4f} "
                  f"d={p['delta_covc']:+.4f} welch p {p['welch_p']:.3g} "
                  f"succ {p['succ_cand_n']}/{p['n']} vs {p['succ_base_n']}/{p['n']} "
                  f"fisher p {p['fisher_p_success']:.3g} ndiff {p['n_seeds_differing']}")
    for k, v in res["H3_invalid_pairing"]["cross_file_paired_vs_champion_same_seeds"].items():
        print(f"  H3 CROSS-FILE {k}: covc {v['covc_cand']:.4f} vs "
              f"{v['covc_base']:.4f} d={v['delta_covc']:+.4f} welch p {v['welch_p']:.3g} "
              f"succ {v['succ_cand_n']}/{v['n']} vs {v['succ_base_n']}/{v['n']} "
              f"fisher p {v['fisher_p_success']:.3g}")
    for k, v in inv.items():
        print(f"  P2 {k}: x {v['x_mean']:.1f} ({v['x_shift_pct_vs_champion']:+.1f}%) "
              f"dyaw_eq {v['equivalent_yaw_shift_deg']:+.2f} deg  "
              f"measured {v['covc_delta_paired_vs_champion']:+.4f}  "
              f"predicted {v['predicted_delta_from_fixture_A_yaw_line']:+.4f}  "
              f"ratio {v['measured_over_predicted']:.3f}")
    print(f"  P3 k*={res['P3_c1_law']['k_star_c1_boundary']:.4f} "
          f"A_max={res['P3_c1_law']['A_max_at_champion_rate_m']:.5f} m, champion uses "
          f"{res['P3_c1_law']['champion_C1_budget_used_frac'] * 100:.1f}% of the budget")
    print(f"  episodes {res['episodes_total']}")


if __name__ == "__main__":
    main()
