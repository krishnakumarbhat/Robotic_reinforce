---
title: "ProgressVLA: Progress-Guided Diffusion Policy for Vision-Language Robotic Manipulation"
arxiv: 2603.27670
venue: arXiv 2026 (cs.RO)
citedByCount: unknown
mechanisms: [progress estimator, differentiable progress guidance, inverse dynamics world model, diffusion policy refinement, CALVIN/LIBERO benchmarks]
cracks: [long-horizon subgoal drift, progress residual 0.07 in sim only, real-world zero-shot unverified for tool substitution, no explicit affordance remap module]
---
ProgressVLA adds task-progress awareness to VLA diffusion policies: a pre-trained progress estimator (residual 0.07 on [0,1]) guides action-token refinement via an inverse dynamics world model. Tested on CALVIN, LIBERO, and real-world deployment. Key gain: terminable long-horizon cascaded tasks without hand-crafted heuristics. Gap for DP-Flow: progress guidance can be fused with affordance-residual norm to terminate tool-improvised chunks when surrogate progress stalls.
Sources: https://arxiv.org/abs/2603.27670
