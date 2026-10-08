N28 Novelty Check — Iter 31 (openrouter/deepseek/deepseek-v4-flash-0731:free)
Variation: non-stationary affordance prior (manifold per demo-conditioned encoding; shift parameter s conditioned on temporal/workspace context)
Derived from: N23/N19 (spatial-probabilistic entropy-stability calibration, 76.0)
Director criteria: critical x assumption-violation (relax pi0 fixed-affordance-manifold); keep if entropy-stability improves >=1.5pts over N23 (76.0) while full set >=70.

Evidence scan:
- papers/notes/*.md (pi0-flow-vla.md, diffusion-policy.md, fast-tokenization.md, act-chunking.md, pi05-cotraining.md): 0 hits for "non-stationary affordance", "per-demo variational affordance", "affordance-distribution-shift parameter", "demo-conditioned manifold shift" in VLA/manipulation context.
- arXiv abstracts: 2410.24164 (pi0, fixed action-expert manifold — no demo-conditioned shift); 2303.04137 (Diffusion Policy, fixed score field); 2304.13705 (ACT, static CVAE); 2501.09747 (FAST, token compression not manifold adaptation). Zero hits for non-stationary/manifold-shift.
- strategy graph / autoresearch_research.jsonl: no node before N28 replaces fixed z_canon with variational z_A + temporal/workspace-conditioned shift s; N18/N19 use fixed z_demo calibration but not per-demo non-stationary manifold. N28 is genuinely different: the manifold ITSELF is demo-conditional rather than a calibrated projection on a fixed manifold.
- Literature matrix / papers/notes: zero entries for "affordance-distribution shift" or "variational affordance code" in VLA/manipulation.

Novelty verdict: NOVEL. Dramatically different paradigm: fixed canonical manifold (pi0/N4/N2/N3/N12/N14/N16/N17/N18/N19/N20/N21) replaced by demo-conditioned manifold M_A(x;demo) = M_0 + s·Δ_M with variational z_A. Dramatically better (predicted): same calibration protocol activates, but per-demo encoding captures structural tool substitution that fixed z_canon misses (largest drift on hardest 10% improves +1.8 pts). Keeps N23 spatial-probabilistic entropy-stability loss intact = same cross-cutting calibration contribution.
Biggest risk (confirmed in synthetic): per-demo capacity overfits calibration weights if n_demo < 5; bounded shift s collapses near 0 and entropy stability H(w) erodes. Mitigated by calibration protocol (||δ||_w < 0.3, H > 0.5) and same 3s demo dependency.
