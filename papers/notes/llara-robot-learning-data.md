---
title: LLaRA: Supercharging Robot Learning Data for Vision-Language Policy
arxiv: 2406.20095
venue: arXiv 2024
citedByCount: unknown
mechanisms: ['LLM-based data augmentation, vision-language policy training, synthetic trajectory generation, robot dataset scaling']
cracks: ['synthetic data quality variance, no contact-aware dynamics, inference overhead from LLM generation']
---
LLaRA uses LLMs to supercharge robot learning data for vision-language policies, generating synthetic trajectories and augmenting real datasets. Enables faster policy training but introduces synthetic bias; contact dynamics remain unmodeled. DP-Flow can use LLaRA-augmented data to train the flow expert without overfitting to synthetic slip patterns.
Sources: https://arxiv.org/abs/2406.20095

