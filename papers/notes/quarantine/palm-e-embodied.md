---
title: PaLM-E: An Embodied Multimodal Language Model
arxiv: 2303.03378
venue: ICML 2023 / arXiv
citedByCount: high (2023-2024)
mechanisms: ['PaLM backbone, embodied sensor fusion, multimodal language reasoning, robot state encoding']
cracks: ['not optimized for real-time control, no action diffusion or chunking, weak on tool improvisation, high compute']
---
PaLM-E fuses PaLM language model with robot sensor inputs for embodied reasoning. Strong on high-level planning but not designed for low-level action diffusion; requires separate policy module. Key limitation for DP-Flow: no flow-matching action expert; affordance reasoning is semantic, not geometric.
Sources: https://arxiv.org/abs/2303.03378

