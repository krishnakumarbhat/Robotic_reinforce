---
title: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
arxiv: 2310.08864
venue: ICRA 2024 (IEEE International Conference on Robotics and Automation)
citedByCount: 1182
mechanisms: [X-embodiment dataset mixture, cross-robot skill transfer, RT-1/RT-2 scale-up, positive-transfer mixture weighting]
cracks: [generalization via data scale not mechanism, no affordance remap for novel tools, mixture weights hand-tuned, contact-rich tool substitution untested]
---
Open X-Embodiment pools robot datasets across embodiments into one mixture and trains RT-X models that show positive cross-robot transfer. Scale (not architectural novelty) drives generalization: shared observation/action normalization lets one policy span manipulators it never saw paired with a given skill. Claim: data diversity substitutes for embodiment-specific tuning. Tasks+data solved: multi-lab mixture, real-robot pickup/move skills across arms. Gap for DP-Flow: transfer is interpolative over the mixture — a structurally divergent improvised tool (banana-as-wipe) is outside every mixture component, so RT-X has no remap path; DP-Flow's per-scene manifold shift + veto gate is the missing mechanism, calibratable from a 3s demo instead of a new mixture slice.
Sources: https://arxiv.org/abs/2310.08864
