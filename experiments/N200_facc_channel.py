"""Purpose: decide whether the DIRECTOR's iter-57 FACC build directive (Force-Adaptive
Contact Control -- an SE(3)/wrench-conditioned action expert replacing a flow-matching
head, validated on "unseen stiffness/geometry" and a "force-threshold success" metric)
has a measurable input channel in the canonical rig, using ONLY already-recorded physical
episodes. No rig byte is read for a knob, no episode is re-simulated, no metric changes.
Inputs: results/aegis_v2/N197_r310_*.jsonl + N197_r310b_*.jsonl + N199_r309_*.jsonl
(two disjoint seed bases, all 3 suites, both paired arms) and
experiments/kaggle_aegis_sweep.py (static prerequisite audit only).
Outputs: prints the prerequisite audit and the per-cell success/failure separation of the
contact-force channel vs non-force controls, and writes results/aegis_v2/N200_facc_channel.json.
"""
import json
import pathlib
import re
import math
import statistics as st

from scipy.stats import mannwhitneyu, spearmanr

ROOT = pathlib.Path("results/aegis_v2")
RIG = pathlib.Path("experiments/kaggle_aegis_sweep.py")
FILES = sorted(ROOT.glob("N197_r310*.jsonl")) + sorted(ROOT.glob("N199_r309_s100_*.jsonl"))
FORCE = ["fn_mean", "fn_p95", "fn_std", "force_compliance", "jerk", "slip_m", "stick_frac", "stall_frac"]
CONTROL = ["friction", "path_len_m", "reg_err_xy_m"]
LEAKAGE = ["coverage_cont_pre"]  # success := coverage_cont >= 0.90, so this is the label itself
NEEDED = ["pi0", "openpi", "flow_match", "flow matching", "action_expert", "vlm", "wrench",
          "stiffness", "force_excess", "energy", "keypoint", "affordance"]


def audit():
    """Purpose: static prerequisite audit of the rig for the FACC build directive.
    Inputs: none (reads RIG text). Outputs: dict of term -> hit count, plus whether
    contactStiffness is a fixed literal or env-driven.
    """
    txt = RIG.read_text()
    low = txt.lower()
    hits = {t: low.count(t) for t in NEEDED}
    knobs = sorted(set(re.findall(r"AEGIS_[A-Z_]+", txt)))
    stiff = [ln.strip() for ln in txt.splitlines() if "CONTACT_K" in ln][:3]
    return {"term_hits": hits, "env_knobs": len(knobs), "contact_k_lines": stiff}


def episodes(path):
    """Purpose: split one rig JSONL into [(level, suite, arm, rows)] using the rig's own
    block order (candidate block then --compare block), the split N197_readout verified.
    Inputs: pathlib path. Outputs: list of (label, rows) with >=1 row.
    """
    rows = [json.loads(ln) for ln in path.read_text().splitlines()
            if json.loads(ln).get("record") == "episode"]
    half = len(rows) // 2
    out = []
    for label, blk in (("dose", rows[:half]), ("frozen", rows[half:])):
        for suite in ("fixture_A", "fixture_B", "fixture_R"):
            out.append((f"{path.stem}|{label}|{suite}",
                        [r for r in blk if r["suite"] == suite]))
    return out


def auc(pos, neg):
    """Purpose: Mann-Whitney separation of the success class from the failure class.
    Inputs: two float lists (pos = successes, neg = failures). Outputs: (auc, p, n_pos, n_neg)
    or None when either class is under 5 episodes.
    """
    if len(pos) < 5 or len(neg) < 5:
        return None
    u, p = mannwhitneyu(pos, neg, alternative="two-sided")
    return (round(u / (len(pos) * len(neg)), 4), float(p), len(pos), len(neg))


def clean(obj):
    """Purpose: replace non-finite floats with null so the artifact is strict JSON.
    Inputs: nested dict/list. Outputs: same structure, NaN/Inf -> None.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def main():
    """Purpose: run the audit + the WITHIN-CELL channel-informativeness test. Pooling across
    cells is confounded (failures concentrate at high pose noise and on round fixtures, where
    every episode statistic shifts), so the honest statistic is the per-cell AUC and its sign
    consistency across cells. Outputs: prints the report, writes N200_facc_channel.json.
    """
    a = audit()
    print("PREREQUISITE AUDIT (rig = %d bytes)" % len(RIG.read_text()))
    for t, n in a["term_hits"].items():
        print("  %-14s hits=%d" % (t, n))
    print("  env knobs: %d ; CONTACT_K defs: %s" % (a["env_knobs"], a["contact_k_lines"]))

    per_cell, devs, collinear = [], {k: [] for k in FORCE + CONTROL}, []
    for path in FILES:
        for label, rows in episodes(path):
            s = [r for r in rows if r["success"]]
            f = [r for r in rows if r["success"] is False]
            if not s or not f:
                continue
            row = {"cell": label, "n_succ": len(s), "n_fail": len(f)}
            for ch in FORCE + CONTROL:
                r = auc([r[ch] for r in s if isinstance(r.get(ch), (int, float))],
                        [r[ch] for r in f if isinstance(r.get(ch), (int, float))])
                row[ch] = r
                if r:
                    devs[ch].append((abs(r[0] - 0.5), r[0] - 0.5, r[1]))
            ok = [r for r in rows if isinstance(r.get("fn_mean"), (int, float))
                  and isinstance(r.get("coverage_cont"), (int, float))]
            rho = spearmanr([r["fn_mean"] for r in ok], [r["coverage_cont"] for r in ok])[0] \
                if len(ok) > 5 else float("nan")
            row["spearman_fn_mean_vs_coverage_cont"] = round(float(rho), 4)
            collinear.append(float(rho))
            per_cell.append(row)
    print("\nCELLS WITH BOTH CLASSES: %d" % len(per_cell))
    for row in per_cell:
        print("  %-58s succ=%3d fail=%3d  fn_mean auc=%s"
              % (row["cell"], row["n_succ"], row["n_fail"],
                 row["fn_mean"][0] if row["fn_mean"] else None))

    print("\nWITHIN-CELL SEPARATION (success vs failure inside one level x suite x arm):")
    res = {}
    for ch in FORCE + CONTROL:
        d = devs[ch]
        if not d:
            res[ch] = None
            print("  %s %-18s no cell with both classes" % ("FORCE" if ch in FORCE else "ctrl ", ch))
            continue
        signs = [1 if s > 0 else -1 for _, s, _ in d]
        consistent = max(sum(1 for x in signs if x > 0), sum(1 for x in signs if x < 0)) / len(signs)
        res[ch] = {"cells": len(d), "mean_abs_dev": round(sum(x[0] for x in d) / len(d), 4),
                   "max_abs_dev": round(max(x[0] for x in d), 4),
                   "sign_consistency": round(consistent, 4),
                   "cells_p_lt_0.01": sum(1 for x in d if x[2] < 0.01)}
        print("  %s %-18s cells=%2d mean|auc-.5|=%.4f max=%.4f sign-consist=%.2f p<.01 in %d"
              % ("FORCE" if ch in FORCE else "ctrl ", ch, len(d), res[ch]["mean_abs_dev"],
                 res[ch]["max_abs_dev"], res[ch]["sign_consistency"], res[ch]["cells_p_lt_0.01"]))
    print("\nCOLLINEARITY WITH THE LABEL: spearman(fn_mean, coverage_cont) per cell,"
          " median |rho| = %.4f over %d cells" % (st.median(abs(x) for x in collinear), len(collinear)))
    print("  LEAKAGE (label itself, excluded as a control): %s" % ", ".join(LEAKAGE))
    verdict = "FACC-INPUT-UNMEASURABLE" if max(
        (r["mean_abs_dev"] for r in res.values() if r), default=0.0) < 0.05 else "FACC-INPUT-PRESENT"
    print("VERDICT: %s" % verdict)
    (ROOT / "N200_facc_channel.json").write_text(json.dumps(
        {"audit": a, "per_cell": clean(per_cell), "within_cell": clean(res), "leakage": LEAKAGE,
         "median_abs_spearman_fn_vs_coverage": round(st.median(abs(x) for x in collinear), 4),
         "verdict": verdict}, indent=2, default=str) + "\n")
    return verdict


if __name__ == "__main__":
    main()
