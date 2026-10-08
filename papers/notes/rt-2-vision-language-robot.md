---
title: RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
arxiv: 2307.15818
venue: Science Robotics / arXiv 2023
citedByCount: very high (500+)
mechanisms: ['web-scale VLM pretraining, robot trajectory fine-tuning, vision-language-action joint model, zero-shot generalization']
cracks: ['web-to-robot transfer weak on contact dynamics, no action chunking, slow real-time inference, no affordance remapping']
---
RT-2 transfers web-scale vision-language knowledge to robotic control via joint fine-tuning. Strong zero-shot object recognition but weak on fine contact dynamics. Uses discrete action tokens from web data; no native diffusion or flow matching. Key gap: DP-Flow replaces discrete tokens with continuous flow chunks and adds contact-aware affordance remap.
Sources: https://arxiv.org/abs/2307.15818

