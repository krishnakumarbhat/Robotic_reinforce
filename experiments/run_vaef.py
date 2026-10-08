"""VAEF: Violation-Aware Affordance-Energy Flow Expert
Director iter 12 — Physical validation complete.
Result: DISCARD (0.2381 << 0.70 keep bar). Freeze N74 92.3.
"""
# Results logged to benchmarks/restroom_sim_result.json
# Key findings:
# - Fixture B with_gate: 0.2381 (keep bar NOT met)
# - Fixture A with_gate: 0.4973 (keep bar NOT met)
# - High variance: seed1=0.9459, seed2=0.2432 (unstable)
# - Root cause: violation-aware energy modulation collapses under real contact
# - FACT-Physics training-procedure alone (N77) showed 0.7297 on Fixture A
# - Architectural energy component + FACT-Physics = collapse
# - Verdict: DISCARD; freeze N74 92.3; same dependency remote GPU deferred
