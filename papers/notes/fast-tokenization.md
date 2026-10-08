---
title: FAST Efficient Action Tokenization for VLA Models
arxiv: 2501.09747
venue: arXiv 2025 (Physical Intelligence)
citedByCount: frontier-2025 (verify at keep time)
mechanisms: [DCT compression, BPE merge, FAST+ universal tokenizer 1M trajectories, autoregressive VLA]
cracks: [750ms/chunk inference vs 100ms diffusion, compression != affordance warp, language-following gain unexplained]
---
FAST: normalize to [-1,1] (1st/99th quantile), per-dim DCT, scale-round prune, interleave low-freq, BPE; invertible, 2 hyperparams. pi0-FAST matches diffusion pi0 incl. laundry folding with 5x fewer GPU-hours; stronger language following. DP-Flow opportunity: FAST tokens as discrete affordance bottleneck for warp module.
Sources: https://www.pi.website/download/fast.pdf , https://huggingface.co/blog/pi0
