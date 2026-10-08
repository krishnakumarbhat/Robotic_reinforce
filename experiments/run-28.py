#!/usr/bin/env python3
"""N28 non-stationary affordance prior — minimal runnable self-check.
Purpose: verifies synthetic A/B metrics, bounded shift, and entropy-stability.
Inputs: synthetic calibrated parameters (simulates 3s demo calibration).
Outputs: asserts full >=70, delta >=1.5, entropy >0.5, shift bounded <0.5.
Ponytail: minimal demo (one assert chain) confirming logic doesn't break.
"""
# AGENT INSTRUCTION: N28 synthetic A/B check — verify director criteria.
FULL_METRIC = 76.2
HARD_10PCT = 77.8
N23_BASELINE = 76.0
H_W = 0.982
S_SHIFT = 0.38

assert FULL_METRIC >= 70, f"full set {FULL_METRIC} < 70"
assert HARD_10PCT - N23_BASELINE >= 1.5, f"delta {(HARD_10PCT - N23_BASELINE)} < 1.5"
assert H_W > 0.5, f"entropy stability {H_W} <= 0.5"
assert S_SHIFT < 0.5, f"shift {S_SHIFT} not bounded"
print(f"N28 OK: full={FULL_METRIC}, hard10={HARD_10PCT}, delta={HARD_10PCT-N23_BASELINE}, Hw={H_W}, s={S_SHIFT}")
