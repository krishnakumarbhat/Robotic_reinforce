"""FACC-SE3 Fixture B 20-seed physical validation.

Director iter 17: Force-Adaptive Contact Control, SE(3)-conditioned,
replaces flow-mat with dynamic affordance field via energy-based
physical in-context attention.

Runs Fixture B / B_height / B_tool with facc_se3 controller,
gate ON and OFF, 20 seeds. Preserves Tier-4 jerk gate.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmarks'))

from restroom_sim import run_suite, FIXTURES, GATE_MAX_JERK, SCRIPTED_BASELINE
import numpy as np

CONTROLLERS = ("facc_se3",)
FIXTURES_TO_TEST = ("fixture_B", "fixture_B_height", "fixture_B_tool")
GATES = (True, False)
SEEDS = range(1, 21)

def main():
    results = run_suite(
        fixtures=FIXTURES_TO_TEST,
        controllers=CONTROLLERS,
        gates=GATES,
        seeds=SEEDS,
        out_path="results/facc_se3_fixture_b_result.json",
        verbose=False,
    )

    summary = results["summary"]
    print(f"\n=== FACC-SE3 Fixture B 20-Seed Validation ===")
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
    with_gate = fixture_b_data.get("facc_se3_with_gate", {}).get("transfer_success")
    without_gate = fixture_b_data.get("facc_se3_without_gate", {}).get("transfer_success")

    print(f"\n=== VERDICT ===")
    verdict = {
        "controller": "facc_se3",
        "fixtures_tested": FIXTURES_TO_TEST,
        "seeds": list(SEEDS),
        "scripted_baseline": SCRIPTED_BASELINE,
        "with_gate": with_gate,
        "without_gate": without_gate,
        "mean_transfer_success": (with_gate + without_gate) / 2 if with_gate is not None else None,
        "keep_bar_met": (with_gate + without_gate) / 2 > 0.70 if with_gate is not None else False,
        "vs_scripted_p": (with_gate + without_gate) / 2 > SCRIPTED_BASELINE if with_gate is not None else False,
        "facc_se3_A_sel_mean": None,
        "facc_se3_energy_curvature_mean": None,
        "facc_se3_force_mean": None,
        "facc_se3_force_peak_mean": None,
    }

    # Extract additional metrics
    for fixture in FIXTURES_TO_TEST:
        fd = summary.get("per_fixture", {}).get(fixture, {})
        for gate_key in ["facc_se3_with_gate", "facc_se3_without_gate"]:
            row_data = fd.get(gate_key, {})
            if row_data:
                verdict[f"facc_se3_{gate_key}_force_peak"] = row_data.get("force_peak_n")
                verdict[f"facc_se3_{gate_key}_force_compliance"] = row_data.get("force_compliance")
                verdict[f"facc_se3_{gate_key}_ctrl_latency_ms"] = row_data.get("ctrl_latency_ms")
                verdict[f"facc_se3_{gate_key}_energy_latency_ms"] = row_data.get("energy_latency_ms")
                verdict[f"facc_se3_{gate_key}_registration_obs"] = row_data.get("registration_obs")
                verdict[f"facc_se3_{gate_key}_attention_entropy"] = row_data.get("attention_entropy")

    with open("results/facc_se3_fixture_b_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nVerdict saved to results/facc_se3_fixture_b_verdict.json")

    # Write detailed row data for analysis
    all_rows = results["rows"]
    facc_rows = [r for r in all_rows if r["controller"] == "facc_se3"]
    with open("results/facc_se3_rows.json", "w") as f:
        json.dump(facc_rows, f, indent=2)
    print(f"Detailed rows saved to results/facc_se3_rows.json ({len(facc_rows)} rows)")

if __name__ == "__main__":
    main()
