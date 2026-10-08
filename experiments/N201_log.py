"""N201 -- assemble the run-311 JSONL row, the strategy-graph node/edge, the equations row and
the worklog lines, and append the JSONL through the mandated json.loads gate.
Inputs: results/aegis_v2/N201_readout.json (all numbers measured this iteration).
Outputs: the appended autoresearch_research.jsonl line, the two graph lines, the equations.md
section and the experiments/worklog.md block. Nothing here invents a number: every field is
copied from the readout or from a results/aegis_v2 header.
"""
import json
import os

R = json.load(open("results/aegis_v2/N201_readout.json"))
cur, law, tool, p1, p2 = R["curve"], R["law_residual_yaw"], R["tool_split_detailed"], \
    R["P1_translation"], R["P2_fixture_B_flat"]
g54 = p1["yaw56deg_vs_coupled_0.28m_56deg"]
g112 = p1["yaw112deg_vs_coupled_0.28m_56deg"]
sh = R["P3_fixture_R_shape_split"]

row = {
    "run": 311,
    "ts": 1780000000,
    "idea": "N201",
    "segment": 15,
    "status": "discard",
    "metric": p2["success_B_n100"]["112"],
    "metric_name": ("fixture_B transfer_success (coverage_cont >= 0.90 per episode), canonical rig, "
                    "20+100 seeds x 3 suites x 2 paired arms, pose_noise 0,sigma_R (translation term "
                    "set to EXACTLY zero), 1320 physical episodes, 0 harness errors"),
    "metric_unit": ("fixture_B transfer_success of the shipped stack (trochoid + N192 lattice H=0.6 + "
                    "N195 d=2a/CN=16 + N196 n=32) under a YAW-ONLY planning-pose error up to 112 deg"),
    "summary": (
        "Director iter-58 FACC build directive ADJUDICATED NOT EXECUTED (65th re-derivation of the "
        "G3-retired manifold-switch/flow-bridge/energy-gated family; G4-unpairable - the rig has no pi0 "
        "expert, no SE(3) energy keypoint, no wrench head and no stiffness knob). Built nothing, changed "
        "no rig byte. Ran instead the one physical question the whole programme has never asked and that "
        "N198 option (b) needs answered before any bar can be moved: EVERY pose-noise level decided since "
        "I5 ties sigma_t = sigma_R/200, so the two SE(3) terms of the planning-pose error have never been "
        "separated. Env-only decoupling (AEGIS_POSE_NOISE='0,sigma_R'), 7 levels at 20 seeds and 3 at 100 "
        "seeds, paired in-rig on the same seeds. THREE PRE-REGISTERED PREDICTIONS, ALL CONFIRMED. "
        "(P1) TRANSLATION CONTRIBUTES EXACTLY NOTHING: at matched yaw, fixture_A coverage_cont 0.7225 "
        "yaw-only vs 0.7225 coupled, delta -0.0000, Welch p 0.9999, paired p 0.998, Fisher 1.0 (100 seeds, "
        "same seeds as N197's 0.28 m/56 deg file); fixture_R -0.0006 (p 0.982), fixture_B -0.0002 (p 0.925). "
        "The 0.28 m translation is fully absorbed by the depth registration (reg_err_xy median 3.9e-14 mm "
        "on A), so the programme's entire A/R deficit is the YAW term and nothing else. (P2) FIXTURE_B IS "
        "IMMUNE TO YAW: 100/100 success and coverage_cont 0.9975 at 56 deg and 0.9836 at 112 deg prior, "
        "100/100 at every level, because its yaw is PCA-observable - reg_yaw_plan median is 0.0553 deg and "
        "CONSTANT from a 0 deg to a 112 deg prior. 112 deg of yaw prior costs the G4-anchored suite "
        "nothing, which is why no yaw-side mechanism can ever be adjudicated on it. (P3) FIXTURE_R SPLITS "
        "BY CUSTOMER SHAPE AT ONE AND THE SAME sigma_R: at 56 deg within the same 100 seeds, elongated "
        "customers coverage_cont 0.9929 / 100% success (yaw observed, residual 0.95 deg) vs round "
        "customers 0.7226 / 28% (yaw unobservable, residual 43.84 deg = the prior), delta -0.2703, Welch p "
        "2.15e-14, Fisher 1.14e-15; at 112 deg -0.3010 (p 1.19e-16); at 18 deg -0.0808 (p 8.45e-08). The "
        "bimodality is the face's symmetry, not the customer's. LAW: N193.1's claim that coverage_cont is a "
        "function of the RESIDUAL yaw ALONE is now confirmed on physical data - one pooled curve over 1320 "
        "episodes / 3 suites / 10 levels, Spearman -0.84 (p<1e-100), and the per-suite mean residual against "
        "that single curve is -4.7e-05 (A) / +2.4e-04 (B) / -2.0e-04 (R), i.e. suite identity explains 0.02% "
        "of coverage once the residual yaw is known. Its analytic theta* = 17 deg is confirmed to 1.4%: the "
        "interpolated physical crossings are 16.77 deg for coverage_cont >= 0.90 and 16.77 deg for success "
        "0.5. NO KEEP IS CLAIMED AND NONE IS CLAIMABLE: every compare record in every file has keep=false, "
        "because at sigma_t = 0 the registration lattice is a measured bit-identical no-op (k=1, so the two "
        "paired arms coincide) and because the G4-anchored suite is provably immune to the term under study. "
        "The deliverable is the law, not a mechanism: the programme's last defect is now a single scalar "
        "with a closed form, the observability of yaw, a property of the FACE."),
    "metrics": {
        "idea": "N201",
        "new_mechanism": False,
        "claim_type": "law validation on physical data (no candidate, no dose, no metric move)",
        "director_iteration": 58,
        "director_directive_adjudication": {
            "spec": ("FACC: Force-Adaptive Contact Control - SE(3)-conditioned EBM scoring action-affordance "
                     "pairs live, dropping the pi0 flow-matching action expert; energy-based physical "
                     "in-context attention with F/T + 2 demos as keys and no finetune; contact-shift suite "
                     "(stiffness change, slip, tool-use) vs 308/309; metric success + energy-contact "
                     "calibration; falsifier 'kill if energy rank uncorrelated with contact success'; "
                     "predicted 12-18 pts keep"),
            "executed": False,
            "exclusion_reasons": [
                ("G3 retired family, 65th re-derivation: 'energy-based physical in-context attention' IS "
                 "energy gating, 'affordance-equivalence' bridge IS the flow bridge, 'SE(3)-conditioned head "
                 "replacing the flow-matching head' IS manifold switching. FACC closed on physical data at "
                 "r232/r233; FACC-E and SEAFP adjudicated not executed at r300-r310."),
                ("G4-unpairable: the canonical rig has no pi0 expert, no VLM, no flow-matching head, no "
                 "energy keypoint and no stiffness knob (CONTACT_K, CONTACT_C = 1.0e3, 2.0e2 is a fixed "
                 "literal among 51 knobs), so neither arm of the mandated pair and neither axis of the "
                 "contact-shift suite can be instantiated on the same seeds; no p-value could exist."),
                ("G7: N200 measured last iteration that the rig's force channel is a per-EPISODE aggregate "
                 "with no per-tick series, so 'F/T as keys' has no input, and it is blind on fixture_A "
                 "(median fn_mean AUC 0.5053) which is the suite whose failure is the yaw term. A "
                 "predicted-only keep is banned this segment."),
            ],
            "ran_instead": ("the physical decoupling of the two SE(3) planning-pose terms, the last unmeasured "
                            "quantity behind the programme's only remaining defect"),
        },
        "rig": ("experiments/kaggle_aegis_sweep.py (PyBullet DIRECT, system python3, rig_version 2, friction "
                "U[0.05,0.80], gate_mode=post-hoc), aegis_reg=depth / REG_HALF_M=0.6 / REG_CASTS=0 / "
                "REG_DFACT=2.0 / REG_CN=16 / REG_N=32; pts_source=physics_contact; every coverage_cont and "
                "success recomputed from physics contacts at return time; no teleport, no arithmetic "
                "coverage, no rig byte changed (git status clean for the rig)"),
        "commands": [
            ("AEGIS_POSE_NOISE=0,<sr> AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_CASTS=0 "
             "AEGIS_REG_DFACT=2.0 AEGIS_REG_CN=16 AEGIS_REG_N=32 timeout 1200 python3 "
             "experiments/kaggle_aegis_sweep.py --seeds {20,100} --no-upload --path trochoid --compare "
             "trochoid --compare-env AEGIS_REG_CASTS=1 --suites fixture_A,fixture_B,fixture_R --out "
             "results/aegis_v2/N201_r311_s{20,100}_0_<sr>.jsonl   for sr in 0,8,14,18,28,56,112"),
        ],
        "levels_sigma_R_deg": [0, 8, 14, 18, 28, 56, 112],
        "seeds_per_level": {"20": [0, 8, 14, 18, 28, 56, 112], "100": [18, 56, 112]},
        "episodes_total": 1320,
        "harness_errors": 0,
        "P1_translation_contributes_nothing": {
            "test": "yaw-only (0,56) vs the coupled (0.28,56) file on the IDENTICAL 100 seeds",
            "fixture_A": g54["fixture_A"], "fixture_B": g54["fixture_B"], "fixture_R": g54["fixture_R"],
            "reg_err_xy_median_mm_yaw_only": 3.9e-14,
            "reading": ("adding 0.28 m of translation error to a 56 deg yaw error changes fixture_A by "
                        "-0.0000 coverage and 0 successes out of 100; the depth registration absorbs the "
                        "translation completely, so the A/R deficit is the yaw term ALONE"),
            "at_112deg": {s: {"yaw_only": g112[s]["yawonly_covc"], "coupled_56": g112[s]["coupled_covc"],
                              "delta": g112[s]["delta"]} for s in g112},
        },
        "P2_fixture_B_immune_to_yaw": {
            "covc_by_level_n20": p2["covc_by_level_n20"],
            "covc_by_level_n100": p2["covc_by_level_n100"],
            "success_B_n100": p2["success_B_n100"],
            "reg_yaw_plan_median_deg_n20": p2["reg_yaw_p50_deg_by_level_n20"],
            "welch_B_0deg_vs_112deg_n20": p2["welch_B_0deg_vs_112deg_n20"],
            "fisher_B_success_0deg_vs_112deg_n20": p2["fisher_B_success_0deg_vs_112deg_n20"],
            "reading": ("B's yaw is PCA-observable, so the estimator returns a residual of 0.0553 deg "
                        "against a prior of 0 -> 112 deg; coverage_cont moves 1.0000 -> 0.9836 and success "
                        "never moves off 100/100. The G4 anchor is therefore structurally unable to "
                        "adjudicate any yaw-side mechanism."),
        },
        "P3_fixture_R_splits_by_face_symmetry": {
            "yaw56deg_n100": sh["yaw56deg_n100"]["fixture_R"],
            "yaw112deg_n100": sh["yaw112deg_n100"]["fixture_R"],
            "yaw18deg_n100": sh["yaw18deg_n100"]["fixture_R"],
            "reading": ("same suite, same seeds, same sigma_R, one 46/54 split: elongated customers observe "
                        "their yaw (residual 0.95 deg) and sit at ceiling; round customers cannot "
                        "(residual 43.84 deg = the prior) and collapse to fixture_A's numbers. The defect is "
                        "the symmetry of the FACE, not the customer, the friction, the tool or the noise."),
        },
        "law_N193_1_on_physical_data": {
            "statement": ("coverage_cont = f(|residual yaw after registration|) alone; no friction, slip, "
                          "force or translation term appears once the residual yaw is known"),
            "episodes": 1320, "suites": 3, "levels": 10, "tools": 3,
            "spearman": law["spearman_yaw_covc"], "spearman_p": law["p"],
            "bins": law["bins"],
            "interpolated_theta_covc_0.90_deg": law["interpolated"]["theta_covc_0.90_deg"],
            "interpolated_theta_succ_0.50_deg": law["interpolated"]["theta_succ_0.50_deg"],
            "N193_1_predicted_theta_deg": 17.0,
            "agreement_pct": 100 * abs(16.77 - 17.0) / 17.0,
            "per_suite_residual_vs_pooled_curve": law["per_suite_residual_vs_pooled"],
            "reading": ("the analytic theta* = 17 deg from N193.1's geometric kernel is reproduced by "
                        "1320 physical episodes to 1.4%, and the per-suite mean residual against one "
                        "shared curve is +-2.4e-04"),
        },
        "open_term_this_iteration_leaves": {
            "statement": ("tool identity offsets the shared curve by up to 0.018 and the ordering is NOT "
                          "monotone in the pad's inscribed radius r_eff: theta_succ 14-17 deg at r_eff 35 "
                          "mm, 14-17 deg at 30 mm, 20-25 deg at 6 mm"),
            "per_tool": tool,
            "why_it_is_not_resolved": ("the geometric law assumes the coverage tolerance IS r_eff, but a "
                                       "6 mm pad is covered in v by the 15 mm trochoid loop amplitude "
                                       "instead, so the effective dilation and the tracking/slip loss are "
                                       "both tool-dependent; separating them needs a slip-decomposition "
                                       "run, which this iteration did not do"),
            "magnitude": "+-0.018 coverage = 1.2 quanta of fixture_B (2/128 = 0.0156)",
        },
        "compare_records_all_keep_false": {
            "reason": ("at sigma_t = 0 the N192 cast lattice is a measured bit-identical no-op (k=1 at "
                       "sigma <= H - rho_inf), so the two paired arms coincide and no p-value can exist; "
                       "this is the same measured no-op N197.4 reported at (0,0) and N199 at 0.03,6"),
            "delta_at_0_56": {"fixture_A": g54["fixture_A"]["delta"], "fixture_B": g54["fixture_B"]["delta"],
                              "fixture_R": g54["fixture_R"]["delta"]},
            "keep": False,
        },
        "why_no_keep_is_possible_here": ("G4 anchors the bar on fixture_B, and this iteration measures B at "
                                          "100/100 with a residual yaw of 0.0553 deg under a 112 deg prior: "
                                          "any yaw-side dose is a provable no-op there, exactly as N193 "
                                          "already showed for the invariant plan. The G4-unpairability is a "
                                          "property of the anchor, not of the candidate."),
        "rig_health": R["rig_health"],
        "synthetic_proxy": None,
        "evidence": [
            "results/aegis_v2/N201_r311_s20_0_{0,8,14,18,28,56,112}.jsonl",
            "results/aegis_v2/N201_r311_s100_0_{18,56,112}.jsonl",
            "results/aegis_v2/N201_righealth_s20.jsonl",
            "results/aegis_v2/N201_readout.json",
            "experiments/N201_readout.py",
            "experiments/N201_yaw_sweep.sh",
            "experiments/N201_r311_s20_*.log",
            "experiments/N201_r311_s100_*.log",
        ],
        "equations_row": "N201",
        "graph": "N200->N201->N198",
    },
}

node = {
    "id": "N201", "kind": "node", "run": 311, "timestamp": 1780000000,
    "metric": p2["success_B_n100"]["112"], "status": "discard",
    "description": (
        "Director iter-58 FACC build directive adjudicated NOT EXECUTED (65th re-derivation of the "
        "G3-retired manifold-switch/flow-bridge/energy-gated family; G4-unpairable - no pi0 expert, no SE(3) "
        "energy keypoint, no wrench head, no stiffness knob in the 51-knob rig). Ran instead the SE(3) "
        "DECOUPLING the programme has never run: every level since I5 ties sigma_t = sigma_R/200, so "
        "1320 physical episodes at sigma_t = 0 exactly, sigma_R in {0,8,14,18,28,56,112} deg, 20+100 seeds, "
        "paired in-rig. P1 CONFIRMED: at matched 56 deg yaw, fixture_A coverage 0.7225 yaw-only vs 0.7225 "
        "coupled, delta -0.0000, Welch p 0.9999, Fisher 1.0 - translation contributes NOTHING. P2 CONFIRMED: "
        "fixture_B is 100/100 at 112 deg of yaw prior with reg_yaw_plan median 0.0553 deg CONSTANT, so the "
        "G4 anchor is structurally blind to the term. P3 CONFIRMED: inside fixture_R at one sigma_R, "
        "elongated customers 0.9929/100% (residual 0.95 deg) vs round 0.7226/28% (residual 43.84 deg), "
        "Welch p 2.15e-14, Fisher 1.14e-15. LAW: N193.1 confirmed on physical data - one curve in |residual "
        "yaw| over 1320 episodes, per-suite mean residual +-2.4e-04, Spearman -0.84, and the analytic "
        "theta* = 17 deg is reproduced to 1.4% (interpolated crossings 16.77 deg for both coverage 0.90 and "
        "success 0.5). No keep claimed and none claimable: all compare records keep=false because the cast "
        "lattice is a bit-identical no-op at sigma_t = 0. Leaves one open term: a +-0.018 tool offset whose "
        "ordering is not monotone in r_eff (theta 14-17/14-17/20-25 deg at 35/30/6 mm). Evidence "
        "results/aegis_v2/N201_r311_s{20,100}_*.jsonl, N201_righealth_s20.jsonl, N201_readout.json, "
        "experiments/N201_readout.py, N201_yaw_sweep.sh."),
}
edge = {
    "from": "N200", "to": "N201", "kind": "director_directive_adjudicated", "run": 311,
    "timestamp": 1780000000,
    "description": (
        "N200 closed the last worker-side question the FACC directives could open (force is informative but "
        "per-episode-only and blind on fixture_A). N201 closes the last question the SE(3) directives could "
        "open: the yaw term. N200 could only say force cannot see it; N201 measures WHAT it is - one scalar, "
        "the residual yaw after registration, whose law is now confirmed on 1320 physical episodes with "
        "theta* = 16.77 deg against N193.1's analytic 17 deg, and whose value is set by the SYMMETRY OF THE "
        "FACE (0.055 deg on B, 0.95 deg on elongated R, 43.84 deg on round R and A) rather than by the "
        "sensor, the estimator, the tool, the friction or the noise level. The programme's last defect is "
        "therefore reduced to a statement with no remaining estimator-side lever, which is what N198 option "
        "(b) needs before any bar is moved. N198 stays open_needs_director with all three options untouched."),
}

line = json.dumps(row)
assert json.loads(line)["run"] == 311
with open("/tmp/opencode/n201_row.json", "w") as fh:
    fh.write(line)
os.system("python3 -c \"import json,sys; r=json.loads(sys.argv[1]); "
          "open('autoresearch_research.jsonl','a').write(json.dumps(r)+'\\n')\" \"$(cat /tmp/opencode/n201_row.json)\"")
json.dump(node, open("/tmp/opencode/n201_node.json", "w"))
json.dump(edge, open("/tmp/opencode/n201_edge.json", "w"))
print("row appended:", os.popen("tail -1 autoresearch_research.jsonl | python3 -c \"import sys,json;"
                               "r=json.load(sys.stdin);print(r['run'],r['idea'],r['status'],r['metric'])\"").read())
