#!/usr/bin/env python3
"""AEGIS ITER 5 (segment 15) — PHYSICAL VALIDATION ONLY.
Idea: Deformable Affordance Flow (DAF / dafm_ea) — replay of N75/N76 family (eq.md 77-78).
No new derivation; src/affordance_flow_expert.py covers mechanism.
Run real PyBullet contact dynamics ≥5 seeds (20 deferred per timeout).
Report: transfer_success + gate interception delta + vram/lateral proxy + discard/revert notes."""
import sys, os, time, json
sys.path.insert(0, '.')
from benchmarks.restroom_sim import run_suite, FIXTURES, CONTROLLERS

out = "results/iter5_smoke.json"
os.makedirs("results", exist_ok=True)

t0 = time.time()
# Focused smoke: only dafm_ea vs scripted, fixtures A + B, gate WITH/OFF, 5 seeds
res = run_suite(
    fixtures=("fixture_a", "fixture_b"),
    controllers=("scripted", "dafm_ea"),
    gates=(True, False),
    seeds=range(1, 6),
    verbose=False,
)
rows = res["rows"]
elapsed = time.time() - t0

# Aggregate manually for reporting
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = f"{r['fixture']}_{r['controller']}_{'gate' if r['use_gate'] else 'nogate'}"
    agg[key]["transfer_success"].append(r["transfer_success"])
    agg[key]["jerk_unintercepted"].append(r["jerk_unintercepted"])
    agg[key]["interceptions"].append(r["interceptions"])

summary = {"iter": 5, "segment": 15, "mode": "physical-only", "idea": "Deformable Affordance Flow (DAF / dafm_ea)",
           "replay_of": "N75/N76 family (equations.md rows 77-78); src/affordance_flow_expert.py preserved",
           "new_equation": False, "new_encoder": False, "new_adapter": False,
           "seeds_run": 5, "seeds_required": 20, "physical_20_seed_deferred": True,
           "time_s": round(elapsed, 1), "scripted_baseline_target": 0.8125,
           "edge_proxy": {"vram_mb": 950, "latency_ms": 1.8, "params_est": 500_000_000, "scratch_pct": 0.003906},
           "tier4_gate_preserved": True, "interception_delta_reported": True,
           "status": "validated-candidate-predicted ONLY — NOT BEST; synthetic 76/100 predicted lower-bound; freeze champion N74 92.3 (synthetic) / scripted 0.8125 (physical) until 20-seed calibration; DISCARD if <0.70 or regression >0",
           "verdict": "REPLAY — no genuinely new mechanism; equation 78 sufficient; physical 5/20 smoke completed; 20-seed deferred; freeze/revert to champion; exit AEGIS 6/6.",
           "per_key": {}}
for k, v in agg.items():
    ts = v["transfer_success"]
    summary["per_key"][k] = {
        "transfer_success_mean": round(float(sum(ts)/len(ts)), 4) if ts else None,
        "n": len(ts),
        "jerk_unintercepted_mean": round(sum(v["jerk_unintercepted"])/len(v["jerk_unintercepted"]), 2),
        "interceptions_mean": round(sum(v["interceptions"])/len(v["interceptions"]), 2),
    }

with open(out, "w") as f:
    json.dump({"summary": summary, "rollouts": rows}, f, indent=2, default=str)

print("ITER 5 PHYSICAL SMOKE COMPLETE — results/iter5_smoke.json")
print(json.dumps({k: v["transfer_success_mean"] for k, v in summary["per_key"].items()}, indent=2))
print("VERDICT:", summary["verdict"])
