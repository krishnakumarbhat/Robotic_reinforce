# Literature Matrix (Phase 1 seed — driver expands)

| Method | Core math | Strength | Failure on tool improvisation | Tier-4 implication |
|---|---|---|---|---|
| ACT (CVAE+ensemble) | chunk VAE + exp ensemble | low-data dexterity | lag + mode collapse on slip | adaptive ensemble weight by affordance residual |
| Diffusion Policy (DDPM/DDIM) | score-field denoise K steps | multimodal, stable train | latency vs contact stability | 5-step straight flow + contact gate |
| pi0 CFM | OT path N(tau A,(1-tau)I), v_theta match u=A-eps | 50Hz dexterous, cross-embodiment | affordance baked in expert | condition field on (c_ctx, phi_afford) |
| pi0.5 co-train | heterogeneous + semantic subtask | unseen-home long-horizon | scale-based, distractible | replace scale with warp + 3s demo |
| FAST | DCT+BPE tokens | 5x train speed, language-follow | slow inference, no warp | FAST tokens as warp bottleneck |
| DP-Flow (ours) | dx/dtau=v_theta(x,tau,c_ctx,phi_aff)+inpaint | zero-shot surrogate tools | — | Tier1-4 500-scenario suite |

Full expansion lives in papers/notes/ + equations.md; driver appends rows per iteration.

| paper | tasks+data solved | metric | gap for DP-Flow |
|---|---|---|---|
| ProgressVLA (2603.27670) | CALVIN/LIBERO long-horizon + real-world | progress residual 0.07, SR up | no affordance remap; add phi_afford to progress regularizer |
| pi0 v4 rev (2410.24164v4) | cross-embodiment fine-tuning, Euler fix | updated expert weights | still no tool-substitution; combine with warp module |
| FAST token (2501.09747) | 1M traj universal tokenizer, 5x train | inference 750ms/chunk | compression artifacts; use tokens only for bottleneck |
| OpenVLA (2406.09246) | multi-robot dataset, open backbone | open weights | insert remap layer into decoder; add progress estimator |
| RT-2 (2307.15818) | web-to-robot zero-shot transfer | high semantic generalization | weak contact dynamics; add flow chunks + contact gate |
| PaLM-E (2303.03378) [UNVERIFIED — S2 rate-limited, recheck] | embodied multimodal reasoning | high-level planning | no low-level action diffusion; use only for c_ctx |
| ACT chunk (2304.13705) | bimanual dexterity, low-data | CVAE ensemble | mode-collapse on slip; weight := f(residual) |
| LLaRA robot (2406.20095) | synthetic data augmentation | faster training | synthetic bias; validate with real contact dynamics |
| Diffusion Policy (2303.04137) | 12-15 tasks, multi-benchmark | +46.9% avg SR | latency vs stability; straight-flow 5-step + gate |
| IEEE survey 11388479 | pi0/FAST/0.5 architecture comparison | survey level | geometric equivalence gap; DP-Flow fills |
| RT-X / Open X-Embodiment (2310.08864) | multi-lab cross-embodiment mixture, real-robot skills | 1182 cites, positive cross-robot transfer | interpolative over mixture; no divergent-tool remap; add per-scene shift + veto gate |
| UMI (2402.10329) | in-the-wild handheld demos, cup/bag/dishwasher tasks | minutes-per-task demos | demos bake tool geometry; 3s demo should buy warp (s, z_scene) + gate, not frozen policy |

### Unverified — recheck when API quota resets (notes in papers/notes/quarantine/)
| NormalFlow (2412.09617) | contact 6DoF tracking | robust under occlusion | tracking only; feed output to phi_afford |
| Predictive inverse (2412.15109) | scalable sim-to-real learners | strong transfer | no VLA conditioning; combine with VLM backbone |
| Dex VLA v2 (2503.06199) | high-DOF finger control | multi-finger chunk | no progress awareness; variable H by residual |
| Gemini robotics (2501.10634) | multi-embodiment long-horizon | semantic reasoning | weak fine control; add flow expert layer |
| Octo? (2405.12213) — S2 429, recheck | — | — | — |
| Mobile ALOHA? (2401.02146) — S2 429, recheck | — | — | — |
| RDT? (2410.07864) — S2 429, recheck | — | — | — |

### QUARANTINED 2026-09-18 — hallucinated arXiv IDs, never cite (notes in papers/notes/quarantine/)
Affordance v3 (2410.04265→linguistics), Contact v2 (2411.01993→perovskite), In-context v2 (2502.05527→graphene), Action token v3 (2504.09178→antennas), Sim-real v4 (2412.01188→Banach), Tool-use VLA (2504.03567→antenna). Caught by verify-before-keep guard.

### QUARANTINED 2026-09-19 batch 2 — ID/title mismatches, never cite
ALOHA-claim (2309.06137→Gaia VMP stars), DP3-claim (2406.05534→Online DPO). Memory-guessed IDs, S2-refuted. Note: papers/notes/quarantine/batch2-id-mismatches.md
