---
title: OpenVLA: An Open-Source Vision-Language-Action Model
arxiv: 2406.09246
venue: arXiv 2024
citedByCount: high (2024-2025)
mechanisms: ['open-source VLA backbone, visual encoding for manipulation, language-conditioned action decoding, multi-robot dataset training']
cracks: ['no built-in progress estimation, limited tool-improvisation without fine-tuning, inference speed lower than diffusion baselines']
---
OpenVLA provides an open-source VLA backbone for vision-language-action control, trained on multi-robot datasets. Enables rapid prototyping of manipulation policies but requires fine-tuning for novel tools. Key benchmark: standard manipulation tasks with multi-embodiment generalization. Gap for DP-Flow: open weights allow direct insertion of affordance remap layer into action decoder; progress estimator can be added post-hoc.
Sources: https://arxiv.org/abs/2406.09246

