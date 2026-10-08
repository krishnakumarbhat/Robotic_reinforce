"""PAITC v2: Physics-Aware Inference-Time Correction (PhysVLA-inspired).

Uses benchmarks/restroom_sim.py run_suite() infrastructure.
PhysVLA (arXiv:2606.13886): phase-aware FSM + selective Euler-Lagrange gate.
Key difference from current paitc (0.2381 FAIL): small blending c=0.05, selective gating.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # Use existing benchmark infrastructure
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
    from restroom_sim import run_suite, FIXTURES, SCRIPTED_BASELINE, CONTROLLERS
    
    # Run PAITC and scripted baseline on Fixture B (zero-shot OOD)
    print("Running PAITC v2 (PhysVLA-inspired) vs scripted baseline...")
    print(f"PhysVLA: c=0.05, selective Euler-Lagrange gate, <1ms target")
    print(f"Current paitc: 0.2381 (FAIL, same as dafm_ea)")
    print()
    
    # The benchmark already has paitc results in benchmark_restroom_sim_result.json
    # PAITC v2 needs to REDESIGN the controller with PhysVLA's small blending
    
    # Load existing results
    with open("benchmarks/restroom_sim_result.json") as f:
        existing = json.load(f)
    
    paitc_existing = {
        "fixture_B_with_gate": existing["fixture_B"]["paitc_with_gate"]["transfer_success"],
        "fixture_B_without_gate": existing["fixture_B"]["paitc_without_gate"]["transfer_success"],
        "energy_latency_ms": existing["fixture_B"]["paitc_with_gate"]["energy_latency_ms"],
    }
    
    print("EXISTING PAITC RESULTS (from benchmark_restroom_sim_result.json):")
    print(f"  Fixture B with gate: {paitc_existing['fixture_B_with_gate']:.4f}")
    print(f"  Fixture B without gate: {paitc_existing['fixture_B_without_gate']:.4f}")
    print(f"  Energy latency: {paitc_existing['energy_latency_ms']:.1f}ms (target <1ms)")
    print()
    
    # PAITC v2 design (not yet run - experiment design):
    v2_design = {
        "controller": "paitc_v2",
        "based_on": "PhysVLA (arXiv:2606.13886)",
        "blend_c": 0.05,  # small blending factor
        "el_gate_epsilon": 0.15,  # selective activation
        "phases": ["approach", "scrub", "rinse", "inspect"],
        "phase_aware": True,
        "selective_gating": True,
        "target_latency_ms": 1.0,  # <1ms per PhysVLA
        "key_design_difference": "Small selective correction (c=0.05) vs current aggressive correction",
        "predicted_fixture_B": "0.0762-0.15 (need to verify)",
        "status": "NOT_RUN - experiment design only",
    }
    
    print("PAITC v2 EXPERIMENT DESIGN:")
    print(json.dumps(v2_design, indent=2))
    print()
    print("CONCLUSION: Current paitc FAILS (0.2381). PhysVLA shows paradigm CAN work")
    print("but current implementation is too aggressive. Redesign needed.")
    print("=" * 60)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/paitc_v2_design.json", "w") as f:
        json.dump({"existing": paitc_existing, "v2_design": v2_design}, f, indent=2)
    
    return v2_design


if __name__ == "__main__":
    main()
