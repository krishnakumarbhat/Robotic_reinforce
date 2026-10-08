# paper1 — Robotics reading corpus (43 papers, basics → frontier)

Downloaded 2026-09-30 from arXiv (all PDFs verified). Read in PART order.
Skild AI publishes NO paper (proprietary) — its lineage is Part 5 (Pathak papers).

## Part 1 — Learning foundations (read first)
| file | paper | why |
|---|---|---|
| 1706.03762.pdf | Attention Is All You Need (Vaswani 2017) | transformers under every VLM/VLA |
| 1312.5602.pdf | DQN (Mnih 2013) | deep RL begins here |
| 1602.01783.pdf | A3C (Mnih 2016) | async actor-critic |
| 1502.05477.pdf | TRPO (Schulman 2015) | trust regions |
| 1707.06347.pdf | PPO (Schulman 2017) | our I3 residual-RL optimizer |
| 1703.03400.pdf | MAML (Finn 2017) | fast adaptation, meta-learning basis |
| 1810.12894.pdf | RND exploration (Burda 2018) | exploration bonuses |
| 2301.04104.pdf | DreamerV3 (Hafner 2023) | world models |

## Part 2 — Robot learning classics
| file | paper | why |
|---|---|---|
| 1504.00702.pdf | End-to-end visuomotor (Levine 2015) | pixels-to-torques proof |
| 1603.02199.pdf | Hand-eye coordination at scale (Levine 2016) | large-scale data collection |
| 1806.10293.pdf | QT-Opt (Kalashnikov 2018) | scalable real-robot RL |
| 2109.12098.pdf | CLIPort (Shridhar 2021) | language-conditioned manipulation |
| 2209.05448.pdf | PerAct (Shridhar 2022) | voxel actions |
| 2304.02643.pdf | SAM (Kirillov 2023) | segmentation backbone |
| 2304.07193.pdf | DINOv2 (Oquab 2023) | visual features |
| 2312.08304.pdf | FoundationPose (Wen 2023) | 6D pose estimation |

## Part 3 — Language → robots
| file | paper | why |
|---|---|---|
| 2204.01691.pdf | SayCan (Ahn 2022) | LLM planning + affordances |
| 2209.07753.pdf | Code as Policies (Liang 2022) | code-writing robots |
| 2307.05973.pdf | VoxPoser (Huang 2023) | open-vocabulary manipulation |

## Part 4 — Modern action policies
| file | paper | why |
|---|---|---|
| 2303.04137.pdf | Diffusion Policy (Chi 2023) | denoising visuomotor SOTA baseline |
| 2304.13705.pdf | ACT / ALOHA (Zhao 2023) | chunking + low-cost hardware |
| 2409.12514.pdf | TinyVLA (Wen 2024) | 422M VLA beating 7B; our edge blueprint |
| 2407.01479.pdf | EquiBot (Yang 2024, CoRL) | SIM(3)-equivariant diffusion |
| 2407.01812.pdf | Equivariant Diffusion Policy (Wang 2024, CoRL oral) | SO(2) symmetry, +21.9% |
| 2405.07503.pdf | Consistency Policy (Prasad 2024) | 10x faster diffusion inference |
| 2410.12557.pdf | Shortcut Models (Frans 2024) | 1-step generation (our I4) |
| 2505.13447.pdf | MeanFlow (Geng 2025) | 1-NFE without distillation (our I4) |

## Part 5 — Skild/Pathak lineage (the video's school)
| file | paper | why |
|---|---|---|
| 1705.05363.pdf | Curiosity-driven exploration (Pathak 2017, ICML) | prediction-error bonus |
| 1906.04161.pdf | Disagreement exploration (Pathak 2019, ICML) | ensemble curiosity |
| 1804.08606.pdf | Zero-shot visual imitation (Pathak 2018, ICLR oral) | imitate from video only |
| 2304.08488.pdf | VRB affordances from human video (Bahl 2023, CVPR) | internet video → robot priors |
| 2207.09450.pdf | WHIRL in-the-wild imitation (Bahl 2022, RSS) | one-shot real-world learning |

## Part 6 — Foundation-model era (frontier)
| file | paper | why |
|---|---|---|
| 2212.06817.pdf | RT-1 (Brohan 2022) | large multi-task real-robot transformer |
| 2307.15818.pdf | RT-2 (Zitkovich 2023) | VLM transfer to robots |
| 2310.08864.pdf | RT-X / Open X-Embodiment (2023) | cross-robot dataset |
| 2406.09246.pdf | OpenVLA (Kim 2024) | open 7B VLA baseline |
| 2410.24164.pdf | pi0 (Black 2024) | flow-matching VLA |
| 2604.15483.pdf | pi0.7 (PI 2026) | diverse context conditioning; our distillation target |
| 2401.02146.pdf | Mobile ALOHA (Fu 2024) | mobile bi-manual, cheap hardware |
| 2602.13193.pdf | Steerable Policies (Chen 2026, RSS) | motion/subgoal steering (our context tokens) |
| 2607.15275.pdf | RoboTTT (NVIDIA 2026) | test-time fast weights, DAgger Distillation |
| 2602.03782.pdf | QVLA (ICLR 2026) | action-centric INT8 (our edge quant story) |
| 2503.14734.pdf | GR00T N1 (NVIDIA 2025) | open humanoid foundation model |
