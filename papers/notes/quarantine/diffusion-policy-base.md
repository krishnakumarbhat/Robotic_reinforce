---
title: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
arxiv: 2303.04137
venue: RSS 2023 / IJRR 2024
citedByCount: 400+
mechanisms: ['conditional DDPM/DDIM, receding-horizon control, FiLM visual conditioning, CNN UNet, K Langevin steps']
cracks: ['DDIM latency vs contact stability, CNN smoothing of velocity, transformer tuning sensitivity, no tool-improvisation']
---
Diffusion Policy learns p(A_t|O_t) via K Langevin steps; achieves +46.9% avg over 12-15 tasks/4 benchmarks. CNN works out-of-box; transformer wins high-freq tasks. DDIM 10-step gives ~0.1s on 3080. Key ablation baseline for DP-Flow latency-vs-stability curve.
Sources: https://arxiv.org/abs/2303.04137

