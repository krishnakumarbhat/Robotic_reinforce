---
title: "Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots"
arxiv: 2402.10329
venue: RSS 2024 (Robotics: Science and Systems Conference)
citedByCount: unknown
mechanisms: [handheld gripper interface, in-the-wild demo collection, fisheye SLAM poses, diffusion policy training, latency-matched deployment]
cracks: [demos still human-driven per scene, no zero-shot tool substitution, gripper-interface morphology fixed, policy inherits demo tool geometry]
---
UMI replaces teleoperation with a handheld gripper: humans demo directly in the wild, SLAM recovers poses, diffusion policies train on the result. Claim: deployment-friction — not model capacity — blocks in-the-wild manipulation. Tasks+data solved: cup/bag/dishwasher tasks from minutes of handheld demos. Gap for DP-Flow: UMI demos bake in the demo tool's geometry — swap the tool and the policy has no equivalence map. DP-Flow's 3s physical demo is the same cost class as a UMI clip but buys a calibrated warp (s, z_scene) plus a veto gate instead of a frozen tool-specific policy.
Sources: https://arxiv.org/abs/2402.10329
