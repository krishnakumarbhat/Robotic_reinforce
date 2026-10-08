"""N201 readout -- yaw/translation decoupling on the canonical rig.

Purpose: quantify how fixture_A / fixture_B / fixture_R coverage_cont and transfer_success
depend on the YAW term of the planning-pose error ALONE, with the translation term set to
exactly zero (AEGIS_POSE_NOISE="0,sigma_R"), and test the three pre-registered predictions
that follow from the already-recorded programme record:
  P1  translation contributes nothing to the A/R deficit: coverage(0,56) ~= coverage(0.28,56)
      on the SAME seeds (N197_r310_s100_0.28,56, restricted to the common seed block);
  P2  fixture_B is FLAT in sigma_R to 112 deg, because its yaw is PCA-observable
      (N193.2: reg_yaw_plan median 0.57 deg against a 41 deg prior);
  P3  fixture_R splits BY CUSTOMER SHAPE at one and the same sigma_R: round customers keep
      the full yaw residual (N193.2: 47.09 deg) and collapse, elongated customers observe
      their yaw (1.05 deg) and stay at ceiling.
Plus the N193.1 law check: coverage_cont is a function of the RESIDUAL yaw alone, so a single
curve in |reg_yaw_plan_deg| must describe every suite and tool.
Inputs: results/aegis_v2/N201_r311_s{20,100}_*.jsonl + the N197 coupled file.
Outputs: results/aegis_v2/N201_readout.json (all numbers quoted in the JSONL row).
"""
import glob
import json
import math
from collections import defaultdict

import numpy as np
from scipy import stats

FILES = sorted(glob.glob("results/aegis_v2/N201_r311_s20_*.jsonl")) + sorted(
    glob.glob("results/aegis_v2/N201_r311_s100_*.jsonl"))
COUPLED = "results/aegis_v2/N197_r310_s100_0.28,56.jsonl"


def load(path, arm=0):
    """Purpose: one arm's episode rows. The rig emits candidate arm then baseline arm."""
    rows = [json.loads(l) for l in open(path) if json.loads(l).get("record") == "episode"]
    per_mode = [json.loads(l) for l in open(path) if json.loads(l).get("record") == "summary_mode"]
    n_arm = per_mode[0]["run"]["episodes"] // len(per_mode) if per_mode else 0
    return rows[:n_arm] if arm == 0 else rows[n_arm:]


def level_of(path):
    """Purpose: the sigma_R (deg) encoded in a file name, e.g. N201_r311_s20_0_56.jsonl -> 56."""
    return float(path[:-6].split("_")[-1])


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def fisher(a, b):
    return float(stats.fisher_exact([[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]])[1])


out = {"episodes": {}, "curve": {}, "P1_translation": {}, "P2_fixture_B_flat": {},
       "P3_fixture_R_shape_split": {}, "law_residual_yaw": {}, "tool_split": {}}
rows_by = {}
for f in FILES:
    lvl = level_of(f)
    rows = load(f)
    n = len(rows)
    out["episodes"][f"{lvl:g}deg_s{20 if '_s20_' in f else 100}"] = n
    rows_by[(lvl, 20 if "_s20_" in f else 100)] = rows

# --- curve -------------------------------------------------------------------------
for (lvl, ns), rows in sorted(rows_by.items()):
    key = f"yaw{lvl:g}deg_n{ns}"
    e = {}
    for s in ("fixture_A", "fixture_B", "fixture_R"):
        sub = [r for r in rows if r["suite"] == s]
        c = [r["coverage_cont"] for r in sub]
        e[s] = {"n": len(sub), "succ": sum(r["success"] for r in sub) / len(sub),
                "covc": float(np.mean(c)), "covc_min": float(np.min(c)),
                "reg_yaw_p50_deg": float(np.median([r["reg_yaw_plan_deg"] for r in sub])),
                "reg_xy_p50_mm": 1e3 * float(np.median([r["reg_err_xy_m"] for r in sub])),
                "reg_ok": sum(r["reg_ok"] for r in sub) / len(sub),
                "esc": sum(r["escaped"] for r in sub) / len(sub)}
    e["keep"] = [json.loads(l)["keep"] for l in open(
        [f for f in FILES if level_of(f) == lvl and (("_s20_" in f) == (ns == 20))][0])
        if json.loads(l).get("record") == "compare"][0]
    e["harness_errors"] = sum(1 for l in open(
        [f for f in FILES if level_of(f) == lvl and (("_s20_" in f) == (ns == 20))][0])
        if json.loads(l).get("status") == "HARNESS-ERROR")
    out["curve"][key] = e

# --- P1: translation contributes nothing (paired, same seeds) -------------------------
coupled = load(COUPLED)
for (lvl, ns) in [(18.0, 100), (56.0, 100), (112.0, 100)]:
    rows = rows_by[(lvl, 100)]
    e = {}
    for s in ("fixture_A", "fixture_B", "fixture_R"):
        cm = {(r["seed"]): r for r in rows if r["suite"] == s}
        cc = {(r["seed"]): r for r in coupled if r["suite"] == s and r["seed"] in cm}
        seeds = sorted(set(cm) & set(cc))
        a = [cm[k]["coverage_cont"] for k in seeds]
        b = [cc[k]["coverage_cont"] for k in seeds]
        t, p = welch(a, b)
        pt = stats.ttest_rel(a, b)
        e[s] = {"n": len(seeds), "yawonly_covc": float(np.mean(a)), "coupled_covc": float(np.mean(b)),
                "delta": float(np.mean(a) - np.mean(b)), "welch_p": p, "paired_p": float(pt.pvalue),
                "yawonly_succ": sum(cm[k]["success"] for k in seeds) / len(seeds),
                "coupled_succ": sum(cc[k]["success"] for k in seeds) / len(seeds),
                "fisher_p": fisher([cm[k]["success"] for k in seeds], [cc[k]["success"] for k in seeds])}
    out["P1_translation"][f"yaw{lvl:g}deg_vs_coupled_0.28m_56deg"] = e

# --- P2: fixture_B flat in sigma_R ---------------------------------------------------
b20 = {lvl: [r for r in rows_by[(lvl, 20)] if r["suite"] == "fixture_B"] for lvl in
       (0.0, 8.0, 14.0, 18.0, 28.0, 56.0, 112.0)}
b100 = {lvl: [r for r in rows_by[(lvl, 100)] if r["suite"] == "fixture_B"] for lvl in
        (18.0, 56.0, 112.0)}
out["P2_fixture_B_flat"] = {
    "covc_by_level_n20": {f"{k:g}": float(np.mean([r["coverage_cont"] for r in v])) for k, v in b20.items()},
    "covc_by_level_n100": {f"{k:g}": float(np.mean([r["coverage_cont"] for r in v])) for k, v in b100.items()},
    "reg_yaw_p50_deg_by_level_n20": {f"{k:g}": float(np.median([r["reg_yaw_plan_deg"] for r in v]))
                                    for k, v in b20.items()},
    "reg_xy_p50_mm_by_level_n100": {f"{k:g}": 1e3 * float(np.median([r["reg_err_xy_m"] for r in v]))
                                    for k, v in b100.items()},
    "welch_B_0deg_vs_112deg_n20": welch([r["coverage_cont"] for r in b20[0.0]],
                                       [r["coverage_cont"] for r in b20[112.0]]),
    "fisher_B_success_0deg_vs_112deg_n20": fisher([r["success"] for r in b20[0.0]],
                                                  [r["success"] for r in b20[112.0]]),
    "success_B_n100": {f"{k:g}": sum(r["success"] for r in v) / len(v) for k, v in b100.items()},
}

# --- P3: fixture_R split by customer shape at ONE sigma_R ----------------------------
split = {}
for (lvl, ns), rows in sorted(rows_by.items()):
    if lvl == 0.0:
        continue
    e = {}
    for s in ("fixture_R", "fixture_A", "fixture_B"):
        byshape = defaultdict(list)
        for r in rows:
            if r["suite"] == s:
                byshape[r["fixture_spec"]["tank_shape"]].append(r)
        if len(byshape) < 2:
            continue
        e[s] = {}
        for shape, sub in sorted(byshape.items()):
            e[s][shape] = {"n": len(sub), "covc": float(np.mean([r["coverage_cont"] for r in sub])),
                           "succ": sum(r["success"] for r in sub) / len(sub),
                           "reg_yaw_p50_deg": float(np.median([r["reg_yaw_plan_deg"] for r in sub])),
                           "reg_yaw_p90_deg": float(np.percentile([r["reg_yaw_plan_deg"] for r in sub], 90))}
        if "round" in e[s] and "elongated" in e[s]:
            rd, el = e[s]["round"], e[s]["elongated"]
            a = [r["coverage_cont"] for r in byshape["round"]]
            b = [r["coverage_cont"] for r in byshape["elongated"]]
            e[s]["round_vs_elongated"] = {
                "delta_covc": rd["covc"] - el["covc"],
                "welch_p": welch(a, b)[1],
                "fisher_p": fisher([r["success"] for r in byshape["round"]],
                                   [r["success"] for r in byshape["elongated"]]),
                "reg_yaw_gap_deg": rd["reg_yaw_p50_deg"] - el["reg_yaw_p50_deg"]}
    split[f"yaw{lvl:g}deg_n{ns}"] = e
out["P3_fixture_R_shape_split"] = split

# --- law check: coverage_cont is a function of |residual yaw| alone ---------------------
allrows = [r for rows in rows_by.values() for r in rows]
yaw = np.array([abs(r["reg_yaw_plan_deg"]) for r in allrows])
cov = np.array([r["coverage_cont"] for r in allrows])
rho, prho = stats.spearmanr(yaw, cov)
edges = [0, 2, 5, 8, 11, 14, 17, 20, 25, 30, 40, 60, 200]
bins = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (yaw >= lo) & (yaw < hi)
    if m.sum() >= 5:
        bins.append({"yaw_lo": lo, "yaw_hi": hi, "n": int(m.sum()),
                     "covc_mean": float(cov[m].mean()), "covc_std": float(cov[m].std()),
                     "succ": float(np.mean([r["success"] for r, k in zip(allrows, m) if k]))})
# per-suite: does the SAME binned curve hold?  (residual of suite - pooled bin mean)
dev = {}
for s in ("fixture_A", "fixture_B", "fixture_R"):
    d = []
    for r in allrows:
        if r["suite"] != s:
            continue
        for b in bins:
            if b["yaw_lo"] <= abs(r["reg_yaw_plan_deg"]) < b["yaw_hi"]:
                d.append(r["coverage_cont"] - b["covc_mean"])
                break
    dev[s] = {"n": len(d), "mean_resid": float(np.mean(d)), "max_abs_resid": float(np.max(np.abs(d))),
              "p90_abs_resid": float(np.percentile(np.abs(d), 90))}
out["law_residual_yaw"] = {
    "spearman_yaw_covc": float(rho), "p": float(prho), "bins": bins, "per_suite_residual_vs_pooled": dev,
    "refuted_note": "N193.1 predicts a single curve; per-suite residuals near 0 confirm it",
}

# --- tool split: theta* should scale with the pad r_eff --------------------------------
ts = {}
for t in sorted({r["tool_id"] for r in allrows}):
    sub = [r for r in allrows if r["tool_id"] == t]
    ok = [r for r in sub if r["success"]]
    bad = [r for r in sub if not r["success"]]
    ts[f"tool{t}"] = {"n": len(sub), "succ": len(ok) / len(sub),
                      "covc": float(np.mean([r["coverage_cont"] for r in sub])),
                      "yaw_p50_fail_deg": float(np.median([abs(r["reg_yaw_plan_deg"]) for r in bad])) if bad else None,
                      "yaw_p50_pass_deg": float(np.median([abs(r["reg_yaw_plan_deg"]) for r in ok])) if ok else None}
out["tool_split"] = ts
out["files"] = FILES + [COUPLED]

with open("results/aegis_v2/N201_readout.json", "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: out[k] for k in ("curve", "P2_fixture_B_flat", "P3_fixture_R_shape_split",
                                      "law_residual_yaw", "tool_split", "P1_translation")}, indent=1)[:7000])
