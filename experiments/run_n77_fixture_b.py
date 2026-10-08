"""N77 FACT-Physics Fixture B 20-seed physical validation.

Runs Fixture B (elongated / wall-hung / matte, ±15cm shift, ±10° yaw)
with fact_phys controller, gate ON and OFF, 20 seeds.
Preserves Tier-4 jerk gate (max_jerk>0.618) WITH and WITHOUT.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmarks'))

from restroom_sim import run_suite, FIXTURES, GATE_MAX_JERK, SCRIPTED_BASELINE
import numpy as np

CONTROLLERS = ("fact_phys",)
FIXTURES_TO_TEST = ("fixture_B", "fixture_B_height", "fixture_B_tool")
GATES = (True, False)
SEEDS = range(1, 21)

def main():
    results = run_suite(
        fixtures=FIXTURES_TO_TEST,
        controllers=CONTROLLERS,
        gates=GATES,
        seeds=SEEDS,
        out_path="results/n77_fixture_b_result.json",
        verbose=False,
    )

    summary = results["summary"]
    print(f"\n=== N77 FACT-Physics Fixture B 20-Seed Validation ===")
    print(f"Total rollouts: {summary['n_rollouts']}")
    print(f"Fixtures tested: {FIXTURES_TO_TEST}")
    print(f"Seeds: {SEEDS}")
    print(f"Scripted baseline target: {SCRIPTED_BASELINE}")
    print()

    for fixture in FIXTURES_TO_TEST:
        fixture_data = summary.get("per_fixture", {}).get(fixture, {})
        print(f"\n--- {fixture} ---")
        for controller in CONTROLLERS:
            for use_gate in GATES:
                key = f"{controller}_{'with_gate' if use_gate else 'without_gate'}"
                fd = fixture_data.get(key, {})
                ts = fd.get("transfer_success")
                if ts is not None:
                    delta = ts - SCRIPTED_BASELINE
                    p_val = "PASS" if ts > 0.70 else "FAIL"
                    print(f"  {key}: transfer_success={ts:.4f}  delta_vs_scripted={delta:+.4f}  keep_bar={'>0.70' if p_val=='PASS' else '<0.70'}  [{p_val}]")
                else:
                    print(f"  {key}: NO DATA")

    # Overall assessment
    fixture_b_data = summary.get("per_fixture", {}).get("fixture_B", {})
    with_gate = fixture_b_data.get("fact_phys_with_gate", {}).get("transfer_success")
    without_gate = fixture_b_data.get("fact_phys_without_gate", {}).get("transfer_success")

    print(f"\n=== VERDICT ===")
    if with_gate is not None and without_gate is not None:
        mean_ts = (with_gate + without_gate) / 2
        if mean_ts > 0.70:
            print(f"KEEP BAR MET: mean transfer_success={mean_ts:.4f} > 0.70")
        elif mean_ts > SCRIPTED_BASELINE:
            print(f"KEEP BAR NOT MET: mean={mean_ts:.4f} > scripted={SCRIPTED_BASELINE} but < 0.70")
        else:
            print(f"DISCARD: mean={mean_ts:.4f} < scripted={SCRIPTED_BASELINE}")
    else:
        print("Incomplete results — check log")

    # Write verdict to file
    verdict_path = "results/n77_fixture_b_verdict.json"
    verdict = {
        "controller": "fact_phys",
        "fixtures_tested": FIXTURES_TO_TEST,
        "seeds": list(SEEDS),
        "scripted_baseline": SCRIPTED_BASELINE,
        "with_gate": with_gate,
        "without_gate": without_gate,
        "mean_transfer_success": mean_ts if with_gate is not None else None,
        "keep_bar_met": mean_ts > 0.70 if with_gate is not None else False,
        "vs_scripted_p": mean_ts > SCRIPTED_BASELINE if with_gate is not None else False,
    }
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nVerdict saved to {verdict_path}")

if __name__ == "__main__":
    main()
