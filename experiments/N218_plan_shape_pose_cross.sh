#!/usr/bin/env bash
# N218 (run 420) -- the PLAN-SHAPE axis CROSSED WITH POSE ERROR.
#
# WHY: run 419 swept the champion's own (A, w) manifold only at pose noise 0,0, where
# `_coverage_cont` is SATURATED (1.0000 for every legal lambda), so that ladder measured the
# C1 legality wall and nothing else. N213 independently measured pose noise as a plan-frame
# TRANSLATION with u as the binder (anisotropy 2.66x on A / 2.61x on B). Nobody has run the
# two axes together, and the loop operator's arithmetic says they must INTERACT:
#   `_loops_offset` (kaggle_aegis_sweep.py:1134): u' = u + A cos(w) - A,  v' = v + A sin(w)
# so the amplitude buys a +/-A margin in v while paying a SYSTEMATIC -A in u. A u-translation
# therefore costs A more headroom, and the u headroom budget is r_eff - A on the high side.
#
# PRE-REGISTERED PREDICTIONS (stated before any run; equations.md ROW N218):
#  Y1 the shape axis is NOT flat under pose error. At 0,0 lambda in [0.05,0.55] all read
#     coverage_cont 1.0000 (zero dynamic range); at 0.03,6 the same ladder must separate, or
#     the plan shape is provably pose-orthogonal and the shape family is closed for robustness.
#  Y2 SIGN TEST on B at 0.03,6: coverage_cont is MONOTONE DECREASING in A over
#     [0.005, 0.030] at fixed lambda = w*A = 0.5, because the binder is u and every +A removes
#     A of the high-u margin r_eff - A. REFUTED if any arm beats the frozen A = 0.015.
#     Prediction is quantitative: d(covc)/dA < 0 with the sign set by (u margin)/(v margin).
#  Y2' the SAME monotone decrease must appear with NO pose noise at all, from a HELD
#     u-offset (AEGIS_POSE_FIX_U = +0.030 m, noise 0,0). Same -A drag, no Gaussian mixture,
#     so the two groups agreeing is a mechanism and disagreeing is a fixture artefact.
#  Y3 w is pose-ORTHOGONAL. At fixed A = 0.015, w only sets the loop rate (loops per metre of
#     base polyline) and never the band EXTENT, so the lambda ladder at frozen A must show no
#     ordered response at 0.03,6. REFUTED if lambda 0.20/0.35/0.55 separate monotonically.
#  Y4 the C1 wall is POSE-INDEPENDENT: lambda_C1 in (0.55, 0.60) is asserted in
#     `scrub_waypoints` (:1195) on the PLAN GEOMETRY, before any physics, so the harness-error
#     count at lambda 0.60 / 0.70 must be IDENTICAL at 0,0 (run 419: A 20/20, R 9/20) and at
#     0.03,6. REFUTED if the noise level changes the tripping count -- that would mean noise
#     reaches plan generation, which would invalidate every noise row in the segment.
#
# CONTROL: `--path trochoid` (frozen champion A=0.015, w=33.3333, lambda=0.5) is arm `a` in
# every file; the ladder point is arm `b` via --compare-env, SAME seeds -> same
# friction/tool/customer/noise draw (G4). No rig byte changed; _coverage_cont VERBATIM; only
# the PLAN moves. gate_mode stays post-hoc. Nothing downloaded.
set -u
cd /media/pope/projecteo/github_proj/a_resume/Robotic_reinforce

run () {  # run <tag> <noise> <amp> <w>
  local tag="$1" noise="$2" amp="$3" w="$4"
  AEGIS_POSE_NOISE="$noise" timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
    --seeds 100 --no-upload --path trochoid --compare trochoid \
    --compare-env "AEGIS_TROCH_AMP_M=$amp,AEGIS_TROCH_W=$w" \
    --suites fixture_A,fixture_B,fixture_R \
    --out "results/aegis_v2/N218_r420_${tag}.jsonl" \
    > "results/aegis_v2/N218_r420_${tag}.log" 2>&1
  echo "done ${tag} rc=$?"
}

# --- GROUP 1 (Y2'): held +0.030 m u-offset, ZERO noise -> pure -A drag, no mixture ---
run fixu030_A005 0,0 0.005 100.0 &
run fixu030_A023 0,0 0.0225 22.222222222222221 &
run fixu030_A030 0,0 0.030 16.666666666666668 &
# --- GROUP 2 (Y1+Y2): Gaussian 0.03,6, lambda = 0.5 fixed, amplitude ladder ---
run n0306_A005 0.03,6 0.005 100.0 &
run n0306_A010 0.03,6 0.010 50.0 &
run n0306_A023 0.03,6 0.0225 22.222222222222221 &
run n0306_A030 0.03,6 0.030 16.666666666666668 &
# --- GROUP 3 (Y3): Gaussian 0.03,6, amplitude FROZEN, w-only ladder ---
run n0306_lam020 0.03,6 0.015 13.333333333333334 &
run n0306_lam035 0.03,6 0.015 23.333333333333332 &
run n0306_lam055 0.03,6 0.015 36.666666666666664 &
# --- GROUP 4 (Y4): Gaussian 0.03,6, C1 wall probe ---
run n0306_lam060 0.03,6 0.015 40.0 &
run n0306_lam070 0.03,6 0.015 46.666666666666664 &
wait
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
echo ALL_DONE