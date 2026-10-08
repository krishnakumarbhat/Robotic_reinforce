---
title: pi0 A Vision-Language-Action Flow Model for General Robot Control
arxiv: 2410.24164
venue: arXiv 2024 (Physical Intelligence)
citedByCount: high-frontier (2024, foundational VLA-flow; verify on Semantic Scholar at keep time)
mechanisms: [PaliGemma VLM backbone, action expert, conditional flow matching, action chunking H=50, cross-embodiment training]
cracks: [needs fine-tuning for novel tools, affordance baked into expert, no ad-hoc tool remap, Euler 10-step bias under contact discontinuity]
---
pi0 fuses PaliGemma VLM with a separate action expert trained by OT conditional flow matching (A^tau=tau*A+(1-tau)*eps, target u=A-eps) on ~10k hours cross-embodiment data; 50Hz chunks. SOTA dexterous baseline our DP-Flow must beat on tool-swap tiers without fine-tuning.
Sources: https://arxiv.org/html/2410.24164v4 , https://www.pi.website/download/pi0.pdf
