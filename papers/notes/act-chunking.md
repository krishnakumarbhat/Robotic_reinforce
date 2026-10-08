---
title: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)
arxiv: 2304.13705
venue: RSS 2023
citedByCount: high (verify at keep time, 500+)
mechanisms: [action chunking, CVAE, temporal ensembling]
cracks: [ensemble decay lag under slip, posterior mode-collapse across affordances, needs demos per tool]
---
ACT chunks H actions from CVAE latent + exponentially-weighted temporal ensemble. Strong low-data baseline; failure mode under tool substitution is our Tier-4 stress test. DP-Flow variation: ensemble weight := f(affordance-residual norm).
