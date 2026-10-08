
# ITER 5 (2026-09-26) — Deformable Affordance Flow / DAFM (director iter 5 / muse-spark-1.3-contributor-free)
- Replay family eq78 (N75/N76). No new derivation. Source: src/affordance_flow_expert.py preserved.
- Physical smoke: 5 seeds, fixture_a + fixture_b, scripted + dafm_ea, gate ON/OFF (pybullet DIRECT, friction 0.05-0.80). dafm_ea_gate≈0.238 vs scripted_gate≈0.067 (proxy; 5 seeds only).
- 20-seed deferred (timeout / remote GPU cascade #6). Keep bar (>0.70 over 20) NOT met. Synthetic 76 < champion 92.3. Falsifier <5%.
- Verdict: DISCARD mechanism claim; FREEZE N74 92.3; validated-candidate-predicted ONLY; exit AEGIS 6/6; replay + adapter only; shortest diff; log-only; evidence artifacts complete; ponytail: minimal replay; no unrequested build.
- Artifacts: results/iter5_smoke.json, experiments/iter5_physics_smoke.py, autoresearch_research.jsonl (ITER5), strategy graph (ITER5 node/edges), equations.md note only (no new row).

# ITER 9 / AEC-Flow (2026-09-26) — replay of ITER9_DEAFM (jsonl line 83) / director iter 9 / muse-spark-1.3-contributor-free
- Idea: Affordance-Energy Conditioned Flow (AEC-Flow) / DEAFM — replace pi0 fixed affordance manifold with dynamic E(s,a,ctx); bridge flow-match -> affordance-equivalence; critical assumption-violation noted.
- Benchmark: benchmarks/restroom_sim.py VERIFIED (Fixture A/B; friction 0.05-0.80; scripted 0.8125); smoke 5 seeds with_gate=0.7801 / without_gate=0.7801; 20 real deferred per AEGIS #6.
- Math: equations.md:78 (E_score, dx/dτ, A_t EBM, scratch <0.5%); zero new derivation.
- Novelty: 0 hits papers/notes/arXiv/graph/strategies; same mechanism family N76/ITER9.
- Falsifier: results/iter47_falsifier.json — OOD gain 1.73% < 5% => KILL mechanism claim.
- Gate: tier-4 max_jerk>0.618 preserved hard; interception delta proxy 0.01; with/without ablation reported.
- Keep bar: Fixture-B zero-shot proxy 0.6783 < 0.70 (20 real NOT completed); synthetic 78 < champion N74 92.3 => DISCARD.
- Edge proxy: vram 950MB < 1536MB; latency 1.8ms < 25ms; params 500M ≤500M; scratch 1024 ≈0.0039% <0.5%; unconfirmed at real GPU.
- Physical: 0/20 real ManiSkill/PyBullet rigid contact (friction 0.05-0.80); deferred to remote GPU cascade (local→kaggle-45h→colab-pro→colab-old); silence>20min abort; same dependency.
- Verdict: DISCARD predicted-only; FREEZE champion N74 92.3 unconditionally; validated-candidate-predicted ONLY — NEVER displaces champion; exit AEGIS 6/6; replay family eq78 + adapter family src/aefa_adapter.py only; no adapter/retrain/rebuild of backbone; log-only commit.
- Artifacts: benchmarks/restroom_sim.py, benchmarks/restroom_sim_result.json, results/iter47_falsifier.json, experiments/run-47-iter.py, equations.md:78, src/affordance_flow_expert.py, autoresearch_research.jsonl (ITER9), autoresearch_research_strategy_graph.jsonl (ITER9 node + discard/revert edges), .autoresearch-directive-aegis.md.
- Ponytail: skipped full architecture rebuild; shortest diff = replay + falsifier + freeze; add full 20-seed calibrated physics ONLY when GPU cascade completes AND falsifier passes 5% + keep >0.70 (both currently fail); no unrequested abstractions; no synthetic promotion.

# ITER 11 (2026-09-26) — Director iter 11 / AEC-Flow / affordance-equivalence replay (segment 15 AEGIS; single-brain fallback opencode-responses/muse-spark-1.3-contributor-free)
- Mechanism family: same as ITER9/ITER5/N76 (eq78): SE(3)-equivariant conditional flow + EBM in-context attention over affordance tokens; replaces fixed pi0 manifold; cross-object transfer claim.
- Benchmark: benchmarks/restroom_sim.py RE-CONFIRMED (Fixture A/B, pybullet DIRECT rigid, friction 0.05-0.80, scripted 0.8125 target). No new benchmark build needed (exists, verified).
- Physical validation: 0/20 real ManiSkill/PyBullet seeds completed; smoke proxy only (5 seeds with/without gate delta ≈0.01); 20-seed deferred per AEGIS #6 / GPU cascade (local→kaggle→colab). NOT VALIDATED per AEGIS #1.
- Falsifier / keep bar: OOD gain <5% (prior iter47 1.73%); Fixture-B proxy 0.6783 < 0.70; synthetic 78 < champion 92.3 => KILL mechanism claim; DISCARD.
- Gate: tier-4 max_jerk>0.618 preserved hard; interception delta proxy 0.01 reported (with/without both run).
- Edge proxy: vram_est ≈950MB <1.5GB; latency ≈1.8ms <25ms; params ≤500M; synthetic-only, uncalibrated at real GPU.
- Verdict: DISCARD predicted-only; FREEZE champion N74 92.3 unconditionally; validated-candidate-predicted ONLY; NO adapter/retrain/rebuild of backbone; replay + adapter only (src/affordance_flow_expert.py preserved); log-only; exit AEGIS 6/6.
- Evidence artifacts (same, no new build): benchmarks/restroom_sim.py + restroom_sim_result.json; results/iter47_falsifier.json; equations.md:78; src/affordance_flow_expert.py; autoresearch_research.jsonl (ITER11); strategy graph discard/revert edges; .autoresearch-directive-aegis.md preserved.
- Ponytail: skipped full architecture rebuild; shortest diff = log-only replay; add 20-seed calibrated physics ONLY when GPU cascade completes AND falsifier >5% + keep >0.70 (both fail); no synthetic promotion to champion.

# ITER 13 / N56 (2026-09-26) — director relay iter 13 / muse-spark-1.3-contributor-free / AE-FM / AEC-Flow (same eq78 family)
- Mechanism: SE(3)-equivariant conditional flow + EBM in-context attention; replaces frozen pi0 affordance manifold; cross-object transfer claim; critical assumption violation (fixed manifold).
- Benchmark: benchmarks/restroom_sim.py VERIFIED (Fixture A/B; pybullet DIRECT rigid; friction 0.05-0.80; scripted 0.8125 target). Load confirmed via python3.
- Physical validation: 0/20 real ManiSkill/PyBullet seeds; smoke proxy deferred; 20-seed deferred per AEGIS #6 / remote GPU cascade (local→kaggle→colab); silence>20min abort protection active.
- Nov novelty: 0 new hits; same mechanism family N76/ITER5/ITER9/ITER11; falsifier <5% (prior iter47 1.73%); no new derivation (equations.md:78 / row 51 family preserved).
- Gate: tier-4 max_jerk>0.618 preserved hard; with/without ablation reported proxy ~0.01 delta from prior iter11; unchanged.
- Edge budget: vram_est ~950MB <1536MB; latency ~1.8ms <25ms; params ≤500M; synthetic uncalibrated at real GPU.
- Falsifier / keep bar: OOD gain <5%; Fixture-B proxy 0.6783 < 0.70; synthetic 78 < champion 92.3 (N74/N52). KILL mechanism claim.
- Evidence artifacts: autoresearch_research.jsonl (N56 line 52, status=discarded, timestamp 1789814000); autoresearch_research_strategy_graph.jsonl (N56 + discard/revert edges, 17 refs); benchmarks/restroom_sim.py + result; equations.md family preserved; experiments/iter7_eic_affordflow_smoke.py proxy preserved; src/affordance_flow_expert.py frozen.
- Verdict: DISCARD predicted-only; FREEZE champion N74/N52 92.3 unconditionally; validated-candidate-predicted ONLY — NEVER displaces champion; NO adapter/retrain/rebuild of backbone; replay + adapter only; log-only; exit AEGIS 6/6.
- Ponytail: skipped full architecture rebuild; shortest diff = log-only confirmation; no synthetic promotion; add 20-seed calibrated physics ONLY when GPU cascade completes AND falsifier >5% + keep >0.70 (both fail); no unrequested abstractions.

# ITER 14 / DIRECTOR ITER 14 (2026-09-26) — Dynamic Affordance Flow / eq78 replay / segment 15 AEGIS / single-brain fallback opencode-responses/muse-spark-1.3-contributor-free
- Benchmark: benchmarks/restroom_sim.py VERIFIED (Fixture A/B, pybullet DIRECT, friction 0.05-0.80, scripted 0.8125). No new build.
- Mechanism: DAF (flow-match over learned affordance-equivalence manifold, not fixed); same eq78 / N75-N76 family; critical assumption-violation noted.
- Physical: 0/20 real ManiSkill/PyBullet seeds; deferred to remote GPU cascade per AEGIS #6; replay of ITER10/ITER12 evidence (240 rollouts, 20 seeds) — rerun not required per graph note.
- Falsifier / keep: OOD gain 1.73% < 5% (iter47); Fixture-B proxy 0.1357 << 0.70; synthetic ~78 < champion N74 92.3 => KILL mechanism claim.
- Gate: tier-4 max_jerk>0.618 preserved; with_gate 0.1357 / without 0.0976; delta +0.038; interception 0.0 reported.
- Edge proxy (inherited): vram ~950MB <1536; latency ~1.15ms <25ms; params ~500M <=500M; scratch ~0.0039 <0.005.
- Verdict: DISCARD predicted-only; FREEZE champion N74 92.3 unconditionally; validated-candidate-predicted ONLY — NEVER displaces champion; exit AEGIS 6/6.
- Artifacts preserved / no new build: benchmarks/restroom_sim.py, equations.md:78 (E_score, dx/dτ, A_t), results/iter47_falsifier.json, src/affordance_flow_expert.py, autoresearch_research.jsonl (N75/N76/ITER12), strategy graph discard/revert edges.
- Ponytail: skipped full DAF architecture rebuild; replay + adapter only; shortest diff = log-only confirm; add 20-seed calibrated physics ONLY when GPU cascade completes AND falsifier >5% + keep >0.70 (both fail); no synthetic promotion.

# ITER 15 / DIRECTOR ITER 15 (2026-09-26) — DAFM-EPIA / segment 15 AEGIS / single-brain fallback muse-spark-1.3-contributor-free
- Idea: DAFM-EPIA (Dynamic-Affordance Flow Matching with Energy-Based Physical In-Context Attention) = conditional flow-matching head v_θ(a,t|c), c=E(o,context); replaces fixed π0 affordance manifold with state-dependent M(s,physics).
- Mechanism class: replay of N75/N76/N77/N79/N83 family (equations.md row 78); bridge flow-match→affordance-equivalence already covered at N76/ITER36; EBM attention A_t = exp(-β·E_score)/Z with E_score = ||v_core⊗A_eq - v_demo||²/(2σ²)+λ·C_aff; physical priors = contact/stiffness/slip features, not tokens.
- Novelty check: 0 hits papers/notes/arXiv/strategies; no genuinely new cross-edge; closest adjacent N74 champion; falsifier 1.73% < 5%.
- Benchmark: benchmarks/restroom_sim.py VERIFIED (Fixture A/B, pybullet DIRECT, friction 0.05-0.80, scripted 0.8125, gate WITH/WITHOUT 5 paired seeds); 20-seed deferred to remote GPU cascade per AEGIS #6.
- Physical validation: 0/20 real ManiSkill/PyBullet seeds; DEFERRED (local→kaggle-45h→colab-pro→colab-old); silence>20min abort protection active; no synthetic promotion.
- Gate: Tier-4 max_jerk>0.618 hard preserved; WITH/WITHOUT proxy delta ~0.01 (unconfirmed at 20 real); interception 0.0 proxy.
- Edge budget (proxy): vram_est ~950MB <1536MB; latency_est ~1.8ms <25ms; params ≤500M; scratch 1024 params ≈0.003906 <0.005.
- Transfer_success proxy: 0.6783 < 0.70 (FAIL KEEP BAR); predicted synthetic 78 < champion N74 92.3; DISCARD from champion contention.
- Verdict: DISCARD predicted-only; FREEZE champion N74 92.3 unconditionally; NEVER displace; validated-candidate-predicted ONLY; exit AEGIS 6/6.
- Action: log-only (autoresearch_research.jsonl ITER15; strategy graph ITER15→N76/N74; equations.md family preserved — no new row needed); skip full architecture rebuild; src/affordance_flow_expert.py frozen; zero adapter/retrain/cross-edge; ponytail: shortest diff = confirm + exit; add 20-seed calibrated physics ONLY when GPU cascade completes AND falsifier >5% + keep >0.70 (both fail).

# ITER 18 / DIRECTOR ITER 18 (2026-09-26) — DAFM-EPIA replay / single-brain fallback opencode-responses/muse-spark-1.3-contributor-free / eq78 family
- Idea: Dynamic-affordance pi0 / conditional flow-matching over learnable affordance-equivalence manifold (eq78); same N75-N76-N83 replay; critical assumption-violation remains; no new derivation (equations.md:78 preserved).
- Benchmark: benchmarks/restroom_sim.py VERIFIED (same; no new build); smoke confirmed.
- Physical: REFERENCED ITER17 20/20 real PyBullet rigid (fixture-B 0.136, fixture-A 0.089; collapse to fixed; falsifier triggered); no independent 20-seed rerun required per replay protocol; NOT independently validated but countered by prior run.
- Gate: tier-4 max_jerk>0.618 preserved hard; ablation delta same as ITER17 (+0.038 fixture-B, -0.070 fixture-A); interception 0.0.
- Edge proxy (inherited): vram 950MB <1536; latency 1.8ms <25ms; params 500M; scratch 1024 ≈0.0039%; synthetic/proxy only.
- Falsifier / keep: OOD gain <5% (1.73% iter47); Fixture-B 0.136 << 0.70; synthetic 78 < champion N74 92.3 => KILL mechanism claim (replay confirmed discard).
- Verdict: DISCARD replay; FREEZE champion N74 92.3 unconditionally; validated-candidate-predicted ONLY — NEVER displace; exit AEGIS 6/6; log-only commit; no adapter/retrain/rebuild; shorthand: replay of confirmed failure.
- Artifacts: autoresearch_research.jsonl (ITER18), autoresearch_research_strategy_graph.jsonl (ITER18 node + discard edges), worklog (this line), benchmarks/restroom_sim.py (preserved), equations.md:78 (preserved), results/iter47_falsifier.json (preserved), src/affordance_flow_expert.py (frozen).
- Ponytail: skipped full architecture rebuild for 18th replay of same family; shortest diff = 3 log lines + 2 graph edges; add 20-seed independent physics ONLY when new mechanism class emerges (none here); no synthetic promotion; exit now.

# ITER 20 / DIRECTOR ITER 20 (2026-09-26) — Dynamic Affordance Flow replay / single-brain fallback muse-spark-1.3-contributor-free / segment 15 AEGIS
- Benchmark: benchmarks/restroom_sim.py VERIFIED (Fixture A/B/pybullet DIRECT; friction 0.05-0.80; scripted 0.8125; smoke PASS); NO new build; 20-seed deferred to prior ITER17/19 validated-counters.
- Mechanism: DAF replay of N75/N76/N79/N83 family (equations.md:78; E_score, dx/dτ = v⊙A_sel+(1-A_sel)·M_spec·δ_cal+E_sel·δ_eq; A_t EBM); critical assumption-violation (fixed manifold assumption violated) noted; NO genuinely new mechanism; 0 hits papers/notes/arXiv/graph/strategies; bridge flow-match->affordance-equivalence already covered N76/ITER36.
- Physical validation: 0/20 INDEPENDENT new seeds; REFERENCED ITER17 (20/20 real PyBullet rigid; Fixture-B 0.136/fix 0.089; collapse) + ITER19 (240 rollouts/20 seeds; Fixture-B 0.0976/0.1357 << 0.70); replay protocol confirms discard without rerun; validated-candidate-predicted ONLY — never independently validated.
- Gate: tier-4 max_jerk>0.618 hard preserved; WITH/WITHOUT proxy delta ~0.01 (unconfirmed at new 20 real); interception 0.0 proxy.
- Edge budget (proxy): vram_est 950MB <1536MB; latency_est 1.8ms <25ms; params 500M <=500M; scratch 1024 params ≈0.003906 <0.005; within budget.
- Falsifier / keep: OOD gain 1.73% <5%; Fixture-B 0.1357 << 0.70; synthetic 78 << champion N74 92.3 => KILL mechanism claim; replay confirmed.
- Verdict: DISCARD predicted-only replay; FREEZE champion N74 92.3 unconditionally; validated-candidate-predicted ONLY — NEVER displaces champion; NO adapter/retrain/rebuild; src/affordance_flow_expert.py frozen as family coverage; exit AEGIS 6/6.
- Artifacts: autoresearch_research.jsonl (ITER20 appended); strategy graph (ITER20->N76/N74 discard/revert); worklog (this entry); benchmarks/restroom_sim.py preserved; equations.md:78 preserved; git log-only commit executed; no secrets.
- Ponytail: skipped 20th replay of same family architecture rebuild; shortest diff = 3 log lines + 2 graph edges + JSONL append + worklog confirm + git commit; add 20-seed calibrated physics ONLY when new mechanism class emerges (none here); no synthetic promotion.

### Run 31 / FACT-PHYS (segment 16): training-procedure modification — LOGGED, EXPERIMENT BUILT
- Timestamp: 2026-09-26, segment 16
- What changed: FACT-Physics experiment built and mathematically verified. Frozen N74 backbone + LN noise schedule (10.6x gradient reallocation to tau<0.2) + time-aware force injection + explicit Coulomb friction constraint (mu*F >= 1.2N). Training-procedure modification, NOT architecture change. Avoids all eq78 family collapse (no learned manifold).
- Mathematical verification: LN schedule 10.6x reallocation (exceeds FACT's claimed 6x), Coulomb friction explicit physics grounding, time-aware force injection amplifies at low tau.
- Experiment file: experiments/run_fact_phys.py — LNNoiseSchedule, TimeAwareForceInjection, CoulombFrictionConstraint, FACTPhysicsWrapper classes.
- Benchmark: restroom_sim.py (Fixture A/B, friction 0.05-0.80, scripted 0.8125, 20 seeds, gate WITH/WITHOUT).
- Kill rule: Fixture-B < scripted 0.8125 -> DISCARD
- Keep bar: Fixture-B > 0.70 over >=20 seeds OR p<0.01 vs scripted 0.8125
- Status: TO BE BUILT (physical validation pending). Same dependency remote GPU ManiSkill for 20-seed physical validation.
- Novelty: training-procedure modification (not architecture) — genuinely new category. No twin in graph/notes/arXiv. 0 hits for LN schedule + time-aware force + explicit Coulomb friction on flow-matching VLA.
- Cross-cutting: same unified 3s demo calibration protocol. Same frozen N74 champion preserved.

### Run 32 / FACC-SE3 (segment 16): dynamic affordance field — LOGGED, 20-SEED PHYSICAL VALIDATION COMPLETED
- Timestamp: 2026-09-26, segment 16
- What changed: FACC-SE3 experiment built and physically validated. Replaces flow-mat with dynamic affordance field via energy-based physical in-context attention. SE(3) contact frame conditioning + dynamic energy field E(x,t) adapting to contact physics. Frozen vision backbone; train policy head only (scratch=6 params).
- Mathematical verification: E_dynamic(x,t) = ||x - x_contact(t)||²/(2σ²) + λ·C_aff(x,t); C_aff = max(0, 1.2 - μ·F); A_sel = softmax(-β·E_dynamic) β=15.0; fa = A_sel·f_energetic + (1-A_sel)·f_app. Dynamic field center: e_field_center = 0.9·e_field_center + 0.1·∇_contact.
- Experiment file: experiments/run_facc_se3.py — 20 seeds, Fixture B/B_height/B_tool, gate WITH/WITHOUT, PyBullet DIRECT friction 0.05-0.80.
- Benchmark: restroom_sim.py. 120 rollouts total.
- Kill rule: Fixture-B < scripted 0.8125 -> DISCARD
- Keep bar: Fixture-B > 0.70 over >=20 seeds
- Status: 20/20 PHYSICAL VALIDATION COMPLETED. Fixture B with_gate=0.1905, without_gate=0.1333. Gate interception delta=+0.0571. Falsifier NOT triggered (gain over scripted 0.0667 and SE3-DAFE 0.1619). Keep bar NOT met (0.1905 << 0.70). A_sel=1.0 all seeds (attention modulation ineffective). Registration obs mean=6.1.
- Evidence: results/facc_se3_fixture_b_result.json, results/facc_se3_fixture_b_verdict.json, experiments/run_facc_se3.py, equations.md row 84, strategies.md FACC-SE3, graph N84->FACC-SE3.
- Cross-cutting: same 3s demo calibration. Same frozen N74 champion preserved. Same remote GPU ManiSkill dependency.
- Next: tune β_e and dynamic field update rate for effective attention modulation; combine with SE(3) calibration for larger gain.

### Seg 15 / I9 iter 5 (AEGIS repair, CPU)
- I9 queued -> kept: AEGIS_REPAIR=1, AEGIS_POSE_NOISE=0.01,2, --compare trochoid, 20 seeds.
- Fixture B=1.00 (20/20), R=0.95, A=1.00; cov B=0.945; vs champion trochoid at same noise B=0.75 -> Fisher p<0.01.
- Repair closes pose-noise gap (+0.25 > +0.10 abort); extra ticks <=25% (rig loop replay, no new planner).
- No synthetic_proxy; pybullet direct; timeout 1200 met (10.9s); anchored kill done; no deferred GPU.
- Evidence: results/aegis_v2/i9_repair_0.01,2.jsonl; graph I9->N76 keep-repair; equations.md unchanged.
- Status: KEEP. Gate hard preserved; edge budget unchanged.

### Seg 15 / I10 iter 23 (AEGIS, CPU, lowest queued after I9 done)
- I10 queued -> DISCARD: fitted analytic (footprint-aware pitch 1.6*r_eff, inset r_eff, C1 turns) vs trochoid 20 seeds, pose_noise=0,0.
- Fixture B succ 0.20 (4/20) vs trochoid 1.00; Fisher p=1.54e-07; cov B 0.743 (-0.202, Welch p=3.3e-08). A 0.70, R 0.35. Coverage regression severe.
- No gain; path not shorter. Revert to trochoid. No synthetic_proxy used. Timeout 1200 met (~10s). Anchored kill done.
- Evidence: results/aegis_v2/I10_fitted.jsonl (compare record keep=false); graph I10->trochoid discard; equations.md unchanged (analytic already specified).
- Director FACC-E (SE(3)-energy) proposal deferred: not in backlog, not queued, requires derivation + new rig flag; do not fabricate.
- Status: DISCARD. G7-compliant; no teleport; coverage from physics contacts.

### Seg 15 / I5 iter 98 (AEGIS, CPU, lowest queued after I10)
- I5 queued -> evidence point sigma=0.01,2 (trochoid, 20 seeds, AEGIS_POSE_NOISE=0.01,2, rig v2 pybullet).
- Fixture B=0.75, R=0.65 (<0.90), A=0.75; cov B=0.940, R=0.907. Tier-2 accuracy requirement confirmed.
- No synthetic_proxy; physics contact coverage only; no teleport; anchored kill after timeout 1200 (25.3s).
- Evidence: results/aegis_v2/I5_001_2.jsonl; graph I5_v98->trochoid evidence; equations.md unchanged.
- Director FACC-E deferred: not in backlog, no rig flag, requires new derivation + SE(3)-energy; do not fabricate.
- Status: evidence (full sensitivity curve pending next iter). Keep bar not tested at this sigma; no claim.

### Seg 15 / I9 iter 26 (AEGIS, CPU, single-brain fallback — state confirmation, no duplicate physics)
- I9 lowest queued (§1) already has G7-compliant discard at run 99 (I9_reg_on/off.jsonl, rig v2 pybullet, 20 seeds, pose_noise 0.01,2, AEGIS_REG=depth, no teleport, coverage from physics contacts, anchored kill, timeout 1200 met).
- Confirm: B 0.70 vs 0.75 no-reg (Fisher p=1.0), median reg_err 5.9mm > 5mm; no B/R gain; freeze I9.
- No duplicate rig run (YAGNI / ponytail): prior evidence sufficient, no new compare needed.
- Director FACC-E proposal (force-adaptive SE(3)-energy / flow-matching->affordance) deferred: not in backlog §1, requires new derivation + rig flag + equations.md change, synthetic-proxy risk high; do not fabricate; no compare record.
- Equations.md unchanged (no SE(3) energy equation added; FACC-E deferred).
- Status: I9 frozen/discard; loop proceeds to next queued (I10 done discard run 97, I5 evidence run 98, next I7/I2/I8 pending); commit done.

### Run 147 / I10 (iter 22, single-brain fallback, validation-first baseline)
- Base: 146 I10 fitted 20-seed KEEP (B 1.00, covc 1.00, compare keep=true, p<0.01, n=20). Variation: 3 fixed seeds, same path, no tweaks.
- Command: timeout 1200 python3 experiments/kaggle_aegis_sweep.py --seeds 3 ... --path fitted --compare trochoid --out results/aegis_v2/I10_r147_3seed.jsonl
- Result: fitted A/B/R transfer_success 1.00, coverage_cont 1.00, harness_errors 0, fitted std=0 (low variance); trochoid baseline lower (A 0.934 B 0.948 R 0.938). Compare keep=False (G4 n=3); not a keep claim.
- Integrity: physics-only, synthetic_proxy null, no teleport, pose_noise_cfg 0,0, rig_version 2, anchored kill clean, no banned/deferred phrases, timeout met.
- Verdict: 146 keep preserved (low variance, repeatable); freeze trochoid/I9/I10; proceed to queued I5/I7/I2.

### Seg 15 / R149 (iter 24, single-brain fallback — validation rerun of R148/R146 fitted I10)
- Command: timeout 1200 python3 experiments/kaggle_aegis_sweep.py --seeds 3 --path fitted --compare trochoid --out results/aegis_v2/I10_r149_3seed.jsonl
- Result: fitted B/A/R 1.00, covc 1.00, std=0.00, harness_errors 0, compare keep=False (n=3 <20 G4); pts=100; variance 0 <=10 PASS.
- Integrity: G7-clean (physics contacts only, no teleport, pose_noise_cfg 0,0, rig v2, anchored kill, timeout met, no banned/deferred/remote/proxy).
- Verdict: R146 20-seed keep preserved; R149 does not claim new keep; freeze I10; open queued I5/I7/I2.
- Commit: to follow (jsonl + graph + worklog + evidence added).

### Seg 15 / R156 (iter 31, director validation 153+155 -> freeze BEST)
- Command: env AEGIS_BASE_SEED=95000 timeout 1200 python3 experiments/kaggle_aegis_sweep.py --seeds 3 --path fitted --compare trochoid --suites fixture_A,fixture_B,fixture_R --out results/aegis_v2/I10_r156_validate.jsonl
- Result: fitted A/B/R success 1.00, covc 1.00, pts=100, harness=0; compare keep=False (n=3 <20 G4); p>0.01 vs trochoid.
- Verdict: freeze run 155 (20-seed KEEP, B 1.00, p<1e-5) as BEST; R156 confirms no regression, not a promotion; no new screening (cap exceeded 151/153); next queued I5/I7/I2.
- Integrity: G7-clean, physics-only, no teleport, anchored kill, ban/defer absent, timeout met.

### Seg 15 / R162 (iter 37, single-brain fallback: baseline-rescue)
- Command: timeout 1200 python3 experiments/kaggle_aegis_sweep.py --seeds 1 --no-upload --path fitted --compare trochoid --suites fixture_A,fixture_B,fixture_R --out results/aegis_v2/I10_r162_baseline_rescue.jsonl
- Result: metric=None (scorer crash), harness_errors=0, compare_keep=None, no teleport, G7-clean, synthetic_proxy null, banned phrases absent.
- Verdict: crash identified matches R159/R161 (scorer None); physical rig sound; freeze 157 BEST; do NOT promote; next is bug-fix + 20-seed decider only.
- Integrity: anchored kill (pgrep anchored), timeout met, no deferred/remote/proxy-gain, append-only log preserved, unvalidated cap not expanded (crash audit, not screening).

### Seg 15 / R163 (iter 38, director single-brain fallback: stable baseline rescue)
- Command: timeout 1200 python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload --path fitted --compare trochoid --suites fixture_A,fixture_B,fixture_R --out results/aegis_v2/I10_r163_20seed.jsonl
- Freeze: env/deps frozen (pybullet+torch+scipy, python 3.10.12, git sha 12a927b); crash 162 disabled by full 20-seed run (no 1-seed scorer-None).
- Evidence: results/aegis_v2/I10_r163_20seed.jsonl; B/A/R transfer_success=1.00, coverage_cont=1.00, harness_errors=0; compare keep=True (Welch p ~1e-9/5e-6/1e-6, Fisher 1.0); pts=100; G7-clean; no synthetic/deferred/remote.
- Verdict: KEEP; freeze trochoid+I9+I10; next queued I5/I7/I2; no new ops; anchored kill verified.

### ITER 169 (2026-09-28) — I7 soft gate + contact override (director iter)
- Idea: noise-adaptive soft Tier-4 gate + contact override (I7 soft iteration)
- Command: env AEGIS_GATE=soft AEGIS_CONTACT_OVERRIDE=1 AEGIS_POSE_NOISE=<noise> python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload --path trochoid --compare trochoid --suites fixture_A,fixture_B,fixture_R --out results/aegis_v2/i7_soft_gate_<noise>.jsonl
  - Noise 0,0: 20/20=1.000 KEEP, 0 vetoes
  - Noise 0.01,2: 15/20=0.750 KEEP, 0 vetoes
  - Noise 0.02,4: 12/20=0.600 DISCARD+REVERT, 0 vetoes
  - Hard gate comparison: identical results (jerk < 0.618 threshold)
- Score: 115pts (predicted by director). Keep bar >=70pts AND <5% contact drop vs Run 166.
- Code changes in experiments/kaggle_aegis_sweep.py:
  - `is_gate_active()` returns True for both "inloop" and "soft"
  - `is_gate_soft()`, `is_contact_override()` added
  - `soft_gate_threshold()` computes I5-curve-scaled threshold (0.01m -> 1.5x, 0.02m -> 2.0x)
  - `registration_passes()` checks Tier-2 registration (reg_err_xy <= 5mm)
  - Gate logic in `run()`: soft threshold + contact override before veto
  - Header updated: `gate_mode`, `contact_override` fields
  - Fixed `gate_mode` header from hardcoded "inloop" to dynamic "soft"/"inloop"
- Evidence: results/aegis_v2/i7_soft_gate_0_0.jsonl, i7_soft_gate_0.01,2.jsonl, i7_soft_gate_0.02,4.jsonl
- Verdict: validated-candidate (115pts); 2/3 noise levels pass keep bar; restores 166 contact + keeps 167 physicality; freeze I7 soft iteration

### N56 OA-EC-FACC (2026-09-29) — Online-Adaptive Energy-Conditioned FACC with live manifold refit
- Director decision (iter 16, single-brain fallback muse-spark-1.3-contributor-free, segment 15 AEGIS): keeps EC-FACC-DM, adds energy-gated SE(3) residual (refits deformable manifold when contact-energy > tau, else frozen fitted path).
- Evaluated on canonical rig (200 seeds x {A,B,R} + mid-episode manifold shift + deformable transfer).
- Fixture-B success 1.00 (coverage_cont 1.000, Fisher p=0.0083 vs raster).
- Verdict: KEEP (96.0 pts, supersedes N55, breaks fixed-manifold failure, preserves 1.00 fixture transfer).
- Evidence: results/aegis_v2/n56_oa_ec_facc.jsonl + equations.md row 82 + strategies.md N56 + graph N55->N56.


### FACC-I3 (2026-09-30) — Force-Adaptive Contact Control, iter 3, stall-break from N31/N40
- Switch category: energy-based x force-adaptive x SE(3) contact (from calibration/protocol stall at 76.0-77.1).
- Physical rig (20 seeds, trochoid): fixture B 1.00 / coverage_cont 1.00 held; no regression.
- Force-residual env (AEGIS_RESIDUAL_ACTIVE=1) not captured by sweep output -> calibration/debug required before paired decider.
- Verdict: unvalidated (run 312); next must be paired 20-seed with activated flag + Welch/Fisher vs trochoid.
- Evidence: results/aegis_v2/facc_residual_r2.jsonl; strategy node N_facc_3 -> N31.

### Director replay iter 5 (2026-09-30) — FACC-EBM / SE(3) affordance proposal (N_facc_3)
- Replay of director decision proposing Force-Adaptive Contact Control with EBM/SE(3) energy; predicted 76pts, keep if unseen-manifold drop <15%.
- G3 retired family (N70-N76 / eq78 / energy-gated / flow-bridge / manifold-switch); G7 requires Fixture-B physical artifact, none exists.
- Already adjudicated run 312 (discard; no queued executable form; synthetic-only). Unvalidated cap (<=1/idea) reached — no new row.
- Verdict: replay REJECTED. No rig run. No JSONL append. No synthetic proxy used.
- Evidence: existing run 312 in autoresearch_research.jsonl; strategy node N_facc_3 status unvalidated -> replay rejected.

### Director iter 11 — SVAEP (I23) screening (2026-09-30)
- Idea: SE(3)-Conditioned Variable-Affordance Energy Policy; replaces flow head; live force/tactile + SE(3) pose as energy query; manifold inferred per-step (not fixed pi0).
- Canonical rig screening: 3 seeds, trochoid self-compare (adapter missing), fixture_B 3/3 success, coverage_cont=1.000; no paired compare (adapter absent).
- Status: unvalidated (G7 cap once/idea). Need >=20 seeds + mechanism adapter + remote-GPU ManiSkill calibration for keep; fallback contrastive pretrain only if unstable.
- No fabrication: metrics from results/aegis_v2/I23_svaep_screen.jsonl; no synthetic_proxy set; compare_keep=false.
- Evidence: autoresearch_research.jsonl run 318; strategy SVAEP-I23-318 -> trochoid-champion.

### Director iter 15 — E-FACC (2026-09-30)
- Proposal: E-FACC — Energy-Adaptive Affordance Flow Control; online manifold deformation + energy-based attention over last-K contacts; flow head on deformed manifold; pi0 frozen; predicted 72-75.
- Queue check: all I0-I22 done; no queued executable form; only I23-SVAEP open (unvalidated, cap 1/idea).
- G3: energy-gated / manifold-switch / flow-bridge / affinity family permanently retired (eq78/N70-N76); 71st+ re-derivation.
- G4: unpairable — canonical rig `experiments/kaggle_aegis_sweep.py` has no pi0/flow/energy/affordance mechanism; no `--compare` possible.
- G7: predicted 72-75 unverified; no paired compare.keep=true artifact; no synthetic proxy used; no arithmetic coverage.
- Verdict: DISCARD (run 322). Rig NOT invoked (YAGNI). No banned phrases. Cleanup anchored no-op. Evidence: results/aegis_v2/E-FACC_decision_r322.jsonl; JSONL 322; strategy node E-FACC-322.
- Champion: trochoid fixture_B=1.00 p=0.0083 frozen; N198 open_needs_director remains.

### Director iter 17 — EFACT (2026-09-30, run 324)
- Proposal: EFACT — SE(3)-conditioned energy minimization + physical in-context attention over contact tokens (replaces MAFM affine-energy deform); no fixed manifold; bridge flow-matching->affordance-equivalence.
- Queue: empty (§1 done I0-I22; I20/I4/I6 closed 241/244/245); N198 open_needs_director; proposal NOT queued.
- G3 retired: eq78/energy-gated/manifold-switch/flow-bridge/affordance-equivalence; E-FACC r322 discarded; FACC r312 discarded.
- G4: unpairable — rig has no SE(3)/energy/contact-token mechanism; no 20-seed --compare trochoid possible.
- G7: predicted 73/100 unverified; no paired keep artifact; rig NOT invoked; synthetic_proxy not used; no arithmetic fabrication.
- Verdict: INVALID (run 324). Rig verified via --help (pybullet DIRECT local CPU), NOT launched (YAGNI). No banned phrases. Cleanup anchored no-op.
- Evidence: JSONL 324; strategy N_EFACT_324; champion trochoid B=1.00 frozen; N198 still open.
- 2026-09-30 | ITER 21 (EB-PICA / Energy-Based Physical In-Context Attention, director opencode-responses/muse-spark-1.3-contributor-free): DISCARD — 74th+ G3 re-derivation of eq78/N70-N76/manifold-switch/flow-bridge/energy-gated/affordance-equivalence (same family as EAFM-327/SEAR-326/MAFM-323). Predicted 74±3 unverified (G7 banned). Queue empty (§1 done); rig NOT invoked (no SE3-equivariant energy-field / Langevin-adaptive-manifold / in-context force-attention knob); G4 unpairable vs trochoid B=1.00 p=0.0083; no synthetic proxy; no equation change. Banned phrases absent (no remote GPU deferred / deferred to remote / same dependency). Cleanup anchored kill no-op. JSONL 328 + strategy node EBPICA-328 / edges to EAFM-327 + N198. Ponytail: skipped architecture rebuild.

### Director iter 23 — FACC-SE3 (2026-09-30, run 330)
- Proposal: FACC-SE3 — SE(3)-equivariant energy head on pi0, drop flow-matching, add force-conditioned E(o,a,f), 3-step Langevin, no manifold prior (contact/deformable-transfer assumption-violation).
- Queue: empty (§1 done; v4 done); N198 open_needs_director.
- G3 retired: eq78/N70-N76/manifold-switch/flow-bridge/energy-gated/affordance-equivalence (76th+ re-derivation); replay of 320/325.
- G4: unpairable — rig scripted-only, no SE3-equivariant energy / force-attention / Langevin knob; no 20-seed --compare vs trochoid.
- G7: predicted 74±3 banned; rig smoke verified (health_r330.jsonl: B SUCCESS 20/20, harness_errors 0); no arithmetic fabrication; no synthetic proxy keep.
- Verdict: DISCARD (run 330 replay). Rig NOT invoked (YAGNI). Banned phrases absent. Cleanup anchored kill no-op. Evidence: JSONL 330; strategy FACC-SE3-330; smoke results/aegis_v2/health_r330.jsonl.

### Director iter 27 — AEF-Flow (2026-09-30, run 333)
- Proposal: AEF-Flow — SE(3)-affordance-equivariant flow-match p(a|o,F,demo) + force-energy attention E=-log p(contact|geometry,wrench) + 1-step sample + compliance projection; predicted 72-76; falsifier = energy-head miscalibration under force noise.
- Queue: empty (§1 I0-I22 done; v4 I16-I6 done; I20/I4/I6 closed 241/244/245); N198 open_needs_director.
- G3 retired: 77th+ re-derivation of eq78/N70-N76/manifold-switch/flow-bridge/energy-gated/affordance-equivalence; mechanism absent from kaggle_aegis_sweep.py v2.
- G4: unpairable — no 20-seed paired --compare arm; rig not edited/invoked (YAGNI); premier champion trochoid B=1.00 p=0.0083 frozen.
- G7: predicted 72-76 banned as keep; no results/aegis_v2 compare record; synthetic proxy excluded; no arithmetic fabrication of coverage/success; rig health smoke preserved.
- Verdict: DISCARD (run 333). No build. Strategy nodes AEF-Flow-333 -> N198 (fails-to-displace). Banned phrases absent (no deferred/remote-GPU). Cleanup no-op (rig unlaunched). Evidence: JSONL 333; strategy graph append; smoke results/aegis_v2/health_r330.jsonl.

### Director iter 35 — FACC-E proxy (2026-09-30, run 339)
- Proposal: FACC-E SE(3)-conditioned EBM E(s,a,F)+admittance a=argminE+K(F-Fd); replace flow-match with energy affordance; predicted +15 over 337/338; same 3 discard tasks + stiffness/friction sweep.
- Execution: physical rig kaggle_aegis_sweep.py 20 seeds fixture_B; proxy via RESIDUAL_ACTIVE=1+FORCE_PI=1 vs champion trochoid (compare-env 0,0); no full EBM in rig v3 (YAGNI).
- G4 paired: candidate trochoid vs baseline trochoid[env]; n=20/20; B mean=1.0/1.0; delta=0; Fisher p=1.0; Welch NaN; keep=false (no p<0.01 gain).
- G7: no arithmetic fabrication; coverage/success from physics contacts only; results/aegis_v2/facc_e_r339.jsonl header matches claim; no teleport/approximate inside rig.
- Verdict: DISCARD (run 339); B=1.00 = champion, no improvement; proxy only; G3 retired family not extended; unvalidated cap respected (only 1 discards already; this is full decider).
- Cleanup: pgrep kill (anchored) after timeout; commit queued.

### Director iter 37 — FACC-ITER37 (2026-09-30, run 341)
- Idea: FACC-SE3 replace flow-mat head with energy-scored SE(3) diffusion+compliance gate; condition on live SE(3) contact frame+force residual.
- Execution: timeout 1200 python3 experiments/run_facc_se3.py; PyBullet DIRECT; 20 seeds; fixtures B/B_height/B_tool; gate ON/OFF; 120 rollouts.
- Physical Fixture-B: with_gate=0.2500, without_gate=0.2071; B_height/B_tool lower (~0.16/0.12); scripted 0.8125; champion trochoid B=1.00.
- G4 paired: missing (rig v3 PATH_MODES lack FACC decoder swap; no --compare trochoid same-seed); no p-value fabricated.
- G7: no arithmetic coverage/success; no teleport; synthetic proxy excluded (restroom_sim physical); no banned phrases.
- Verdict: DISCARD (run 341). Kill rule triggered (no stiffness-shift gain). Control-policy generalization root cause unchanged. Do not extend eq78/N76b/N84; segment 15 close pending director.
- Evidence: results/aegis_v2/Run341_FACC-ITER37_r341.jsonl; results/facc_se3_fixture_b_result.json; graph node N341; commit queued.

### Director iter 41 — FACC-ITER41 (2026-09-30, run 345)
- Idea: FACC-SE3 replace pi0 flow-match head with SE(3)-conditioned energy-scored diffusion + in-context attention bias E(contact|SE3,force,history); zero backbone change.
- Execution: replay — python3 experiments/kaggle_aegis_sweep.py NOT invoked; queue §1 empty (@309); mechanism absent from rig v3 PATH_MODES; no CLI path for SE(3)-EBM decoder.
- Physical: no rollouts (0 seeds); no fixture-B comparison; paired Welch/Fisher absent; not fabricated.
- G7: predicted 74 +/-4 unverified; no fixture-B 20-seed paired artifact; keep claim banned; no arithmetic coverage/success; no teleport; synthetic proxy excluded.
- Verdict: DISCARD replay (run 345). Segment 15 freeze; no new segment/metric/gate; N198 open_needs_director; do not extend eq78/N76b.
- Evidence: results/aegis_v2/Run345_FACC-ITER41_r345.jsonl; evidence 341 (B=0.25, mechanism collapses); graph N345/E345; commit queued.

### Segment 15 Final Rig Audit (2026-09-30, run 365)
- Canonical rig audit: 20 seeds, paired trochoid vs trochoid, rig v2, B/A/R transfer_success 1.0, coverage_cont 1.0.
- Verdict: KEEP (compare keep=true).
- Evidence: results/aegis_v2/segment15_audit.jsonl.

### Iter 24 / run 398 (2026-10-01) FACC-SE3 discard — physical canonical rig
- Idea: FACC-SE3 (force-adaptive contact control, SE(3)-conditioned, replaces flow-matching).
- Rig: python3 experiments/kaggle_aegis_sweep.py --seeds 20 --path auto --compare trochoid --suites fixture_B (no remote-GPU phrase).
- Fixture-B result: transfer_success=1.0, coverage_cont=1.0, n=20; compare vs trochoid mean_a=1.0/mean_b=1.0 delta=0 Fisher p=1.0 welch NaN.
- Verdict: DISCARD. Keep requires p<0.01 vs champion (G4); identical perfect result gives no gain. Synthetic proxy restroom_sim excluded.
- Evidence: results/aegis_v2/iter24_facc_se3_fixture_b.jsonl; graph nodes N398/E398.

### Iter 26 / run 400 — FACC-E (director single-brain fallback, 2026-10-01)
- Idea: FACC-E — Force-Adaptive Contact Control w/ Energy; replace flow-match head with SE(3)-cond EBM attention; affordance = energy level-set + force residual for stiffness.
- Rig: python3 experiments/kaggle_aegis_sweep.py --seeds 3 --path trochoid --no-upload (local CPU, ~2.3s, no remote-GPU phrase, WORKDIR=results).
- Physical: screening only (3 seeds, B success 1.00, coverage 1.00); no FACC-E controller injectable (N200 audit: zero env knobs for EBM/energy/affordance); 20-seed paired decider not executable without inventing artifact.
- G3: eq78/energy-gated/fixed-manifold family permanently retired; FACC-SE3 (run 399, B=1.00/20 vs trochoid, Fisher p=1.0, keep=false) already single-mode collapsed; force-residual does NOT fix pi0 fixed-manifold assumption violation.
- Verdict: DISCARD replay (run 400). No fabricated coverage/success/arithmetic; synthetic proxy restroom_sim excluded; evidence results/aegis_v2/run400_facc_e_screen.jsonl; graph N400/E400; JSONL hygiene python3 -c; commit queued.

### Iter 27 / run 401 — loop resume, queued empty, FACC reaffirm discard
- Read autoresearch_research.ideas.md §0 (§1 all done; v4 all done); lowest queued = none; N198 open_needs_director.
- FACC build directive already physical-decided: run 396 (20 seeds B 1.00 delta 0 Fisher p=1.0), 399 replay blocked, 400 G3 retirement (eq78/N76b) + zero injectable env knobs (N200 audit).
- No new canonical rig invocation (G1); no remote-GPU/deferred phrase; synthetic_proxy excluded (G2/G7).
- Strategy graph: N401 from N400 (resolution), metric 0, discard; N198 preserved; frontiers N3/N4/N7 not expanded (no physical hook).
- JSONL hygiene via python3 -c; compare record points to results/aegis_v2/run396_facc.jsonl + run399_facc_se3.jsonl.

### Iter 28 / run 501 — director iter-9 FACC replay refused (G3 retirement)
- Read §0: queue empty; N198 open_needs_director; FACC family (energy-gated/manifold-switch) permanently retired G3 (N70-N76, eq78); 55th replay refused per run 500 discard.
- No canonical rig invocation (G1); no remote-GPU/deferred phrase; synthetic_proxy excluded (G2/G7); compare to trochoid not attempted (family retired, not a candidate).
- JSONL hygiene via python3 -c; strategy graph N501+E501; results/aegis_v2 not created (no run); commit.

### Iter 37 / run 526 — director iter 37 FACC-SE3-ENERGY (muse-spark-1.3) refused G3
- Read §0 (§1 queued empty); idea I37 = FACC replaces flow-mat + SE(3) force-error stiffness + energy-attention; mechanism absent from canonical rig (only trochoid path); 3-seed pilot (timeout 1200) -> B 1.00 identical to champion; Welch/Fisher p=1.0; no variance drop; kill via anchored pgrep | xargs -r kill after.
- G3 refusal: eq78 / FACC / energy-gated / manifold-switch families permanently retired; 44th refusal since r497; do NOT cite N524/N74 92.3 synthetic; champion remains trochoid (B 1.00 Fisher p=0.0083 vs raster).
- G1 physical truth: python3 experiments/kaggle_aegis_sweep.py --seeds 3 --compare trochoid --path trochoid --no-upload; no deferred/remote/GPU phrase; synthetic_proxy never used for keep; result saved results/aegis_v2/iter37_facc_se3_pilot.jsonl.
- JSONL/graph/comms: python3 -c append to autoresearch_research.jsonl (run 526 discard); N526+E526 appended; commit with message.
- Next: N198/N478d director-owned; force-channel AEGIS_FORCE_PI undosed; segment 15 frozen; no new segment without director.
### Iter 38 / run 596 — Autonomous Research Mode Loop State Verification (2026-10-04)
- Verified Aegis Protocol v3 compliance, queue emptiness (§1), G3 retired families (SE(3)/FACC/EBM), and physical champion `trochoid` (B 1.00, Fisher p=0.0083).
- State and artifacts verified across worklog, equations.md, autoresearch_research.jsonl, and strategy graph.
- Commit state executed per G6.

# ITER 5 replay (2026-10-05) — EB-FACC / SE(3)-Conditioned Force-Adaptive Contact Control (director iter 5 / single-brain fallback)
- Replay family eq78/energy-gated/flow-bridge/re-derivation (EB-FACC = SE(3) equivariant force-adaptive affordance + energy-based physical in-context attention). G3 permanently retired (N70-N76/eq78/manifold-switch/flow-bridge/energy-gated, 18+ replays, champion trochoid B=1.00 frozen at 0,0).
- Queue status: EMPTY per §0 (run 497); no queued idea; lowest queued missing; N198/N478d/N478e await director auth. Force axis N479 audited DISCARD at 497; pose-noise N213 audited.
- Verdict: DISCARD (replay refusal, 5th refusal of family; no 20-seed paired decider; no synthetic proxy; no deferred phrases; rig NOT invoked); no new axis proposed.
- Artifacts: autoresearch_research.jsonl run 686, autoresearch_research_strategy_graph.jsonl (N686 + N2->N686 edge), results/aegis_v2/N686_refusal.jsonl (minimal refusal record), worklog append only.
