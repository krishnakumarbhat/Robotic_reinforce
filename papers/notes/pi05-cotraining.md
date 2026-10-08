---
title: pi0.5 a Vision-Language-Action Model with Open-World Generalization
arxiv: CoRL 2025 PMLR 305:17-40
venue: CoRL 2025
citedByCount: frontier-2025 (verify on Semantic Scholar at keep time)
mechanisms: [heterogeneous co-training, high-level semantic subtask prediction, knowledge insulation, web data + verbal instruction]
cracks: [generalization via data scale not geometric equivalence, subtask inference distractible, modest context/memory, fails unmodeled surrogates]
---
pi0.5 co-trains mobile-manipulation (400h) with other-robot + lab + web/caption/QA/localization data; hierarchical: predict semantic subtask then low-level chunk. First end-to-end long-horizon (10-15min) unseen-home cleanup. Our target: same generalization WITHOUT 400h per-embodiment, via affordance warp + 3s demo.
Sources: https://proceedings.mlr.press/v305/black25a.html , https://www.pi.website/download/pi05.pdf
