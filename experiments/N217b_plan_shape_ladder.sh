#!/usr/bin/env bash
# N217b (run 419) -- complete the PLAN-SHAPE ladder that the orphaned N217 arms left open.
# Every arm: one canonical-rig invocation, candidate = frozen champion (A=0.015, w=33.3333,
# lambda=0.5) paired IN-RIG against the ladder point via --compare-env on the SAME 20 seeds (G4).
# Nothing is downloaded; gate_mode stays post-hoc; _coverage_cont VERBATIM (only the PLAN moves).
set -u
cd /media/pope/projecteo/github_proj/a_resume/Robotic_reinforce
run () {  # run <tag> <amp> <w>
  local tag="$1" amp="$2" w="$3"
  timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
    --seeds 20 --no-upload --path trochoid --compare trochoid \
    --compare-env "AEGIS_TROCH_AMP_M=$amp,AEGIS_TROCH_W=$w" \
    --suites fixture_A,fixture_B,fixture_R \
    --out "results/aegis_v2/N217b_r419_${tag}.jsonl" \
    > "results/aegis_v2/N217b_r419_${tag}.log" 2>&1
  echo "done ${tag} rc=$?"
}
# lambda ladder at frozen A=0.015 (w = lambda/A).  0.05/0.20/0.35 probe the LOW side the
# orphaned run never reached; 0.55/0.70 bracket the (0.5, 0.8] cliff; 0.0225@lambda=0.5
# brackets the A cliff between 0.020 (clean on B) and 0.030 (B 0.9625).
run lam005 0.015 3.3333333333333335 &
run lam020 0.015 13.333333333333334 &
run lam035 0.015 23.333333333333332 &
run lam055 0.015 36.666666666666664 &
run lam070 0.015 46.666666666666664 &
run amp0225 0.0225 22.222222222222221 &
wait
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
echo ALL_DONE