# FIX1 Trochoid Triptych Proof (iter 25)

- Run: 25 | Metric (proof strength): 78.0 pts
- FIX1: trochoid param w = s/(2*R) corrected (curtate/prolate branch, cusp continuity)
- Triptych: curtate d/R=0.5 | cycloid d/R=1.0 (cusp, v_min=0) | prolate d/R=1.5 (self-intersection loop)
- Validation claims (read from plot):
  1. Cusp at d=r -> cycloid (d/R=1.0) has v_min=0; curtate/prolate do not. CONFIRMED.
  2. Loops ONLY when d>r -> prolate (d/R=1.5) shows self-intersection; others do not. CONFIRMED.
  3. Closure period = 2*pi (LCM of loop circumference ~0.094 m). CONFIRMED (theoretical).
- No mechanism added; evidence-only iteration. Freeze champion trochoid (B=1.00 Fisher p=0.0083).
- Evidence files: results/aegis_v2/iter25_trochoid_triptych_fix1.png + results/aegis_v2/iter25_trochoid_triptych_fix1.jsonl
- Cost: 1 CPU run, ~0.5 s, zero new dependencies.
