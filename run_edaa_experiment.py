#!/usr/bin/env python3
"""EDAA execution wrapper — minimal paired rig run (iter 32, director proposal).
Runs EDAA-deformed conditioning manifold against frozen trochoid champion
(20 seeds, fixture_A/B/R, same friction/tool/noise/customer, G4 paired).
Physical only; no synthetic proxy for keep claim.
Kill rules enforced: discard if fixture_B transfer_success <= 0.70 or no
significant gain vs champion (Welch p >= 0.01 or Fisher p >= 0.01), or if EBM
attention_uniformity >= 0.95 (uniform collapse).
"""
import os, sys, time, json, math, random, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply EDAA mechanism to fixture spec/manifold before passing to rig
from experiments.run_edaa import edaa_condition
from experiments.kaggle_aegis_sweep import FIXTURES, run_episode, summarize, welch

SEEDS = 20
SUITES = ["fixture_A", "fixture_B", "fixture_R"]
MODE = "trochoid"
OUT = "results/aegis_v2/edaa_iter32.jsonl"

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

results = {"candidate": [], "champion": []}
start = time.time()

# Run candidate: EDAA deforms conditioning manifold (fixture spec offset/manifold)
# We apply the SE(3) field to the fixture offset (manifold deformation) and log
# the EBM energy/uniformity per episode.
with open(OUT.replace(".jsonl", "_cand.log"), "w") as logf:
    for suite in SUITES:
        spec = FIXTURES.get(suite, FIXTURES.get("fixture_A", {}))
        for k in range(SEEDS):
            seed = 91000 + 1000 * (SUITES.index(suite) * SEEDS) + k
            # Apply EDAA conditioning manifold deformation
            deformed_offset, energy, uniformity, field = edaa_condition(spec, seed=seed)
            # Modify fixture spec's offset/manifold (not teleport — G7 compliant)
            edaa_spec = dict(spec)
            edaa_spec["offset_cm"] = round(deformed_offset[0] * 100, 2)
            edaa_spec["offset_y_cm"] = round(deformed_offset[1] * 100, 2)
            edaa_spec["angle_deg"] = spec.get("angle_deg", 0) + math.degrees(field[3])
            # Minimal synthetic proxy check banned; physical rig only
            try:
                rec = run_episode(suite, edaa_spec, seed, "pybullet", mode=MODE)
                rec["edaa_energy"] = energy
                rec["edaa_uniformity"] = uniformity
                rec["edaa_field"] = list(field)
                rec["edaa_manifold_shift_m"] = math.hypot(field[0], field[1])
                results["candidate"].append(rec)
                logf.write(json.dumps({"seed": seed, "suite": suite,
                                      "tag": rec.get("quality_tag"),
                                      "success": rec.get("success"),
                                      "coverage_cont": rec.get("coverage_cont"),
                                      "energy": energy, "uniformity": uniformity,
                                      "field": list(field)}) + "\n")
                logf.flush()
            except Exception as exc:
                # G5: crash -> log, commit, exit (do not fabricate)
                logf.write(f"CRASH seed={seed} suite={suite}: {type(exc).__name__}: {str(exc)[:140]}\n")
                logf.flush()
                # Exit with crash status; no synthetic fill-in
                raise SystemExit(f"EDAA crash at seed={seed}: {exc}")

# Run champion (same seeds, same mode, frozen manifold — no deformation)
with open(OUT.replace(".jsonl", "_champ.log"), "w") as logf:
    for suite in SUITES:
        spec = FIXTURES.get(suite, FIXTURES.get("fixture_A", {}))
        for k in range(SEEDS):
            seed = 91000 + 1000 * (SUITES.index(suite) * SEEDS) + k
            try:
                rec = run_episode(suite, spec, seed, "pybullet", mode=MODE)
                results["champion"].append(rec)
                logf.write(json.dumps({"seed": seed, "suite": suite,
                                      "tag": rec.get("quality_tag"),
                                      "success": rec.get("success"),
                                      "coverage_cont": rec.get("coverage_cont")}) + "\n")
                logf.flush()
            except Exception as exc:
                logf.write(f"CHAMPION CRASH seed={seed} suite={suite}: {type(exc).__name__}: {str(exc)[:140]}\n")
                logf.flush()
                raise SystemExit(f"Champion crash at seed={seed}: {exc}")

# Aggregate and paired compare (G4)
for suite in SUITES:
    a = [r["coverage_cont"] for r in results["champion"]
         if r.get("suite") == suite and str(r.get("status", "")).startswith("PHYSICAL")]
    b = [r["coverage_cont"] for r in results["candidate"]
         if r.get("suite") == suite and str(r.get("status", "")).startswith("PHYSICAL")]
    sa = [bool(r.get("success")) for r in results["champion"] if r.get("suite") == suite and str(r.get("status", "")).startswith("PHYSICAL")]
    sb = [bool(r.get("success")) for r in results["candidate"] if r.get("suite") == suite and str(r.get("status", "")).startswith("PHYSICAL")]
    cmp = welch(a, b)
    cmp["suite"] = suite
    cmp["n_a"] = len(a)
    cmp["n_b"] = len(b)
    cmp["succ_a"] = sum(sa)
    cmp["succ_b"] = sum(sb)
    # Fisher exact on binary success
    try:
        from scipy.stats import fisher_exact
        table = [[int(sum(sb)), int(len(sb) - int(sum(sb)))],
                 [int(sum(sa)), int(len(sa) - int(sum(sa)))]]
        cmp["fisher_p"] = float(fisher_exact(table)[1])
    except Exception:
        cmp["fisher_p"] = None
    # Kill rules (G4/G7): keep requires fixture_B transfer_success > 0.70 AND p < 0.01 vs champion
    fb = {"n_b": len(b), "transfer_success": round(sum(sb)/max(1,len(b)), 4),
          "mean_coverage_cont": round(sum(b)/max(1,len(b)), 4),
          "compare": cmp}
    # EDAA-specific kill: uniform attention collapse
    unif_vals = [r.get("edaa_uniformity", 0) for r in results["candidate"]
                 if r.get("suite") == "fixture_B"]
    unif_mean = sum(unif_vals) / max(1, len(unif_vals))
    fb["edaa_uniformity_mean"] = round(unif_mean, 4)
    fb["edaa_collapse_kill"] = bool(unif_mean >= 0.95)
    fb["kill_uniform_attention"] = unif_mean >= 0.95
    # Compute verdict
    n_b = fb.get("n_b", 0)
    ts_b = fb.get("transfer_success", 0.0)
    sig = (cmp.get("welch_p", 1.0) < 0.01 and cmp.get("delta", 0) > 0) or \
          ((cmp.get("fisher_p") or 1.0) < 0.01 and sum(sb) > sum(sa))
    keep = bool(n_b >= 20 and ts_b > 0.70 and sig and cmp.get("delta", -1) >= -0.005)
    fb["verdict"] = "KEEP" if keep else ("DISCARD" if fb.get("edaa_collapse_kill") else "DISCARD+REVERT")
    fb["status"] = "UNVALIDATED" if n_b < 20 else fb["verdict"]
    # Never claim keep without evidence file; log artifact reference
    fb["evidence_file"] = OUT
    # Log summary line (JSONL hygiene: valid JSON line only)
    result_line = {
        "run": 32, "segment": 15, "idea_id": "EDAA",
        "status": fb["status"], "verdict": fb["verdict"],
        "fixture_B_transfer_success": fb.get("transfer_success"),
        "fixture_B_coverage_cont_mean": fb.get("mean_coverage_cont"),
        "compare_p_welch_coverage": cmp.get("welch_p"),
        "compare_p_fisher_success": cmp.get("fisher_p"),
        "edaa_uniformity_mean": fb.get("edaa_uniformity_mean"),
        "edaa_collapse_kill_triggered": fb.get("edaa_collapse_kill"),
        "n_seeds_fixture_B": fb.get("n_b"),
        "champion": "trochoid", "evidence_artifact": OUT,
        "timestamp": time.time(), "metric_class": "physical"
    }
    with open(OUT, "w") as fh:
        fh.write(json.dumps(result_line) + "\n")
    print(json.dumps(result_line, indent=2))

    # Append-only worklog update (G6, G7)
    worklog_path = "experiments/worklog.md"
    with open(worklog_path, "a") as wf:
        wf.write(f"\n### Iter 32 EDAA (EDAA) — director proposal\n")
        wf.write(f"- Timestamp: 2026-09-28, segment 15 (closed physical regime)\n")
        wf.write(f"- Proposal: Energy-Deformed Affordance Attention — not controller, manifold-only\n")
        wf.write(f"- Mechanism: 2k-param EBM E(o,h) -> SE(3) equivalence field, frozen flow-mat\n")
        wf.write(f"- Physical validation: paired vs trochoid (20 seeds, fixture_A/B/R)\n")
        wf.write(f"- Fixture B: n={fb.get('n_b',0)}, transfer_success={fb.get('transfer_success',0):.4f}\n")
        wf.write(f"- Compare: Welch p={cmp.get('welch_p', 'N/A')}, Fisher p={cmp.get('fisher_p','N/A')}\n")
        wf.write(f"- EDAA uniformity mean: {fb.get('edaa_uniformity_mean', 'N/A')} (collapse if >=0.95)\n")
        wf.write(f"- Verdict: {fb['verdict']}; kill triggers: uniform_attention={fb.get('kill_uniform_attention', False)}\n")
        wf.write(f"- Evidence: {OUT}\n")
        wf.write(f"- Note: no synthetic proxy used; all claims tied to physical {OUT}\n")
        wf.write(f"- Status: {fb['status']} (unvalidated allowed cap checked: {fb['status'] == 'unvalidated'})\n")

    # Update equations.md if mechanism equation changed (minimal: note row)
    eq_path = "equations.md"
    if os.path.exists(eq_path):
        with open(eq_path, "a") as ef:
            ef.write(f"\n### Row EDAA (iter 32, director proposal)\n")
            ef.write(f"- EBM E(o,h) -> SE(3) field; lightweight ~2k params; global lock noted.\n")
            ef.write(f"- Manifold deformation: delta = tanh(W_field * z) * scale; no teleport (G7).\n")

    # Strategy graph node update (minimal)
    graph_path = "autoresearch_research_strategy_graph.jsonl"
    graph_node = {"node_id": "EDAA", "derived_from": ["N84", "N76b"],
                  "category": "energy-based", "status": fb["status"],
                  "metric": fb.get("transfer_success"), "segment": 15,
                  "evidence": OUT}
    with open(graph_path, "a") as gf:
        gf.write(json.dumps(graph_node) + "\n")

    # Git commit every iteration (G6)
    subprocess.run(["git", "add", "experiments/worklog.md", "equations.md",
                    "autoresearch_research_strategy_graph.jsonl", OUT,
                    "autoresearch_research.jsonl", "experiments/run_edaa.py"],
                   capture_output=True)
    commit_msg = f"EDAA iter32: {fb['verdict']} B={fb.get('transfer_success', 'N/A')} p={cmp.get('welch_p', 'N/A')}/f={cmp.get('fisher_p', 'N/A')} uniformity={fb.get('edaa_uniformity_mean', 'N/A')}"
    subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True)

    # Final kill-rule enforcement and cleanup (G5)
    # Anchored kill pattern (not bare pkill)
    import signal
    # After run: kill any leftover python3 sweep processes with anchored pattern
    subprocess.run("pgrep -f '^python3 .*kaggle_aegis_sweep\.py' | xargs -r kill",
                   shell=True, capture_output=True)

    print(f"EDAA iter32 completed in {time.time() - start:.1f}s. Verdict={fb['verdict']}. Evidence={OUT}")
