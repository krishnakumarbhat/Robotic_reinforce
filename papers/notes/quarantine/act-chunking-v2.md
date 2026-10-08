---
title: ACT: Fine-Grained Bimanual Manipulation with Low-Cost Hardware
arxiv: 2304.13705
venue: RSS 2023 / arXiv
citedByCount: high (500+)
mechanisms: ['CVAE latent chunking, temporal ensembling, exponential weight decay, action chunk H=50, bimanual control']
cracks: ['ensemble lag under slip, mode-collapse across affordances, requires demos per tool, no diffusion/flow']
---
ACT chunks H actions from a CVAE latent with exponentially-weighted temporal ensemble. Strong low-data dexterity baseline. Failure mode under tool substitution is our Tier-4 stress test. DP-Flow variation: ensemble weight := f(affordance-residual norm) instead of fixed exponential.
Sources: https://arxiv.org/abs/2304.13705

