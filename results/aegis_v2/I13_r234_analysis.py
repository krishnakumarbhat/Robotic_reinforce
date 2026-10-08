#!/usr/bin/env python3
"""I13 decider: read the two 20-seed paired rig files, emit the verdict JSON.

Purpose: turn results/aegis_v2/I13_r234_a{4.0,8.0}_k001.jsonl into the numbers the
worklog + JSONL row cite, so no claim in the log has to be retyped by hand (G7).
Inputs: nothing (paths are the two rig files next to this script). Outputs: I13_r234_result.json.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
KEYS = ("mean_coverage_cont", "escaped_frac", "mean_launch_frac",
        "mean_z_exc_max_m", "mean_slip_m", "transfer_success")
# I13 pre-registered bars (ideas.md I13): escapes < 5% AND fixture_B success >= 0.50.
# G4 keep bar: >=20 seeds, B > 0.70, Welch p<0.01 (coverage) or Fisher p<0.01 (success).


def read(path: pathlib.Path) -> dict:
    """Purpose: split a rig JSONL into header / compare / per-arm summary.
    Inputs: file. Outputs: {header, compare, arms: {label: per_suite}}.
    """
    rows = [json.loads(line) for line in path.open()]
    head = rows[0]
    cmp = next(r for r in rows if r.get("record") == "compare")
    arms = {r["path_mode"]: r["per_suite"] for r in rows if r.get("record") == "summary_mode"}
    return {"header": head, "compare": cmp, "arms": arms}


out = {"idea": "I13", "run": 234, "verdict": "DISCARD (pre-registered abort fired)",
       "bar": {"escapes_lt": 0.05, "fixture_B_success_ge": 0.50,
               "keep_B_gt": 0.70, "keep_p_lt": 0.01},
       "mechanism": ("v(s) = v0/(1+alpha*kappa(s)) realised as a tick->arclength TIME "
                     "re-parameterisation; kappa = |d theta|/ds of the horizontal heading; "
                     "total tick budget unchanged; no force/solver/scoring term touched"),
       "flat_champion_ref": {"file": "results/aegis_v2/v2_trochoid_0,0.jsonl",
                             "fixture_B_success": 1.0, "fixture_B_coverage_cont": 0.9453,
                             "fixture_B_slip_p90_m": 0.0106},
       "configs": {}}
for alpha in (8.0, 4.0):
    p = HERE / f"I13_r234_a{alpha}_k001.jsonl"
    d = read(p)
    c = d["compare"]
    cand_label = [k for k in d["arms"] if "SPEED_ALPHA=0" not in k][0]
    base_label = [k for k in d["arms"] if "SPEED_ALPHA=0" in k][0]
    out["configs"][f"alpha={alpha}"] = {
        "file": str(p.relative_to(HERE.parent.parent)),
        "header_check": {k: d["header"].get(k) for k in
                         ("seeds_requested", "surface", "bowl_k_m", "speed_alpha",
                          "pose_noise_cfg", "path_mode", "rig_version")},
        "candidate_arm_knobs": c["candidate_knobs"].get("SPEED_ALPHA"),
        "baseline_arm_knobs": c["baseline_knobs"].get("SPEED_ALPHA"),
        "keep": c["keep"],
        "per_suite": {s: {
            "n": c[s]["n_b"],
            "success_base": c[s]["succ_a"], "success_cand": c[s]["succ_b"],
            "coverage_cont_base": c[s]["mean_a"], "coverage_cont_cand": c[s]["mean_b"],
            "coverage_cont_delta": c[s]["delta"],
            "welch_p": c[s].get("welch_p"), "paired_p": c[s].get("paired_p"),
            "fisher_p": c[s].get("fisher_p"),
            "arm_base": {k: d["arms"][base_label][s][k] for k in KEYS},
            "arm_cand": {k: d["arms"][cand_label][s][k] for k in KEYS},
        } for s in ("fixture_A", "fixture_B", "fixture_R")},
    }

# regression: the frozen rig must be bit-identical at alpha=0
def eps(name: str) -> list:
    """Purpose: episode rows of a regression JSONL. Inputs: file name. Outputs: rows."""
    return [r for r in (json.loads(x) for x in (HERE / name).open())
            if r.get("record") == "episode"]


pre, post = eps("I13_r234_regression_alpha0_PRE-edit.jsonl"), eps("I13_r234_regression_alpha0_POST-edit.jsonl")
k = ("seed", "success", "coverage", "coverage_cont", "slip_m", "jerk", "stall_frac",
     "stick_frac", "path_len_m", "steps")
out["frozen_rig_regression"] = {
    "episodes": len(pre),
    "fields_compared": list(k),
    "bit_identical": len(pre) == len(post) and not [1 for x, y in zip(pre, post)
                                                    for f in k if x.get(f) != y.get(f)],
    "additive_fields_only": sorted(set(post[0]) - set(pre[0])),
}
(HERE / "I13_r234_result.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k2: v for k2, v in out.items() if k2 != "configs"}, indent=1))
for a, c in out["configs"].items():
    b = c["per_suite"]["fixture_B"]
    print(f"{a}: B {b['success_base']}/{b['n']} -> {b['success_cand']}/{b['n']} "
          f"covc {b['coverage_cont_base']} -> {b['coverage_cont_cand']} "
          f"(d={b['coverage_cont_delta']}, welch_p={b['welch_p']:.3f}, fisher_p={b['fisher_p']}) "
          f"escapes {b['arm_base']['escaped_frac']} -> {b['arm_cand']['escaped_frac']} "
          f"launch {b['arm_base']['mean_launch_frac']} -> {b['arm_cand']['mean_launch_frac']} "
          f"slip {b['arm_base']['mean_slip_m']} -> {b['arm_cand']['mean_slip_m']} keep={c['keep']}")
