# FIX1 Analytic Proof + Arc-Length Closure (Run 26)

- Fixed R=0.015m; d/R sweep [0.5, 1.0, 2.0] (curtate / cycloid / prolate)
- Analytic velocity: v = √(R²+d²−2Rd·cosθ) overlaid on plot
- Cusp confirmation: cycloid v_min=0.0; curtate=0.0075; prolate=0.015
- Arc-length: polyline L=0.120000m vs 8R=0.120m; error=0.0000%; pass=True
- Topology: curtate loop=False area=0.001237; prolate loop=True area=0.004948
- Status=validated-candidate; metric=84.0; discard=False; reasons=[]
- Evidence: results/aegis_v2/iter26_trochoid_fix1_analytic.jsonl + results/aegis_v2/iter26_trochoid_fix1_analytic.png + plot image
