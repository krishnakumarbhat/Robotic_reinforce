---
title: Diffusion Policy Visuomotor Policy Learning via Action Diffusion
arxiv: 2303.04137
venue: RSS 2023 / IJRR 2024
citedByCount: 400+ (IJRR 2024 lists 348; RSS version 421 — verify live at keep time)
mechanisms: [conditional DDPM/DDIM, receding-horizon control, FiLM visual conditioning, time-series diffusion transformer, CNN UNet]
cracks: [DDIM latency vs contact stability, CNN over-smoothing of velocity commands, transformer tuning sensitivity, no tool-improvisation]
---
Diffusion Policy: p(A_t|O_t) via K Langevin steps; +46.9% avg over 12-15 tasks/4 benchmarks. CNN works out-of-box; transformer wins high-freq tasks. DDIM 100->10 gives 0.1s on 3080. Key ablation baseline for DP-Flow latency-vs-stability curve.
Sources: https://arxiv.org/html/2303.04137v4 , https://doi.org/10.1177/02783649241273668
