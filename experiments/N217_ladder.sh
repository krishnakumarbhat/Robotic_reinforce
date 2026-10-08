#!/usr/bin/env bash
# N217 ladder driver. Every arm: paired in-rig against the frozen champion on the SAME 20 seeds
# via --compare-env, canonical rig, timeout 1200, --no-upload. Nothing downloaded.
set -u
cd /media/pope/projecteo/github_proj/a_resume/Robotic_reinforce
run () {  # run <tag> <amp> <w> [noise]
  local tag="$1" amp="$2" w="$3" noise="${4:-0,0}"
  AEGIS_POSE_NOISE="$noise" timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
    --seeds 20 --no-upload --path trochoid --compare trochoid \
    --compare-env "AEGIS_TROCH_AMP_M=$amp,AEGIS_TROCH_W=$w" \
    --suites fixture_A,fixture_B,fixture_R \
    --out "results/aegis_v2/N217_r366_${tag}.jsonl" \
    > "results/aegis_v2/N217_r366_${tag}.log" 2>&1
  echo "done ${tag} rc=$?"
}
for spec in "$@"; do
  IFS=: read -r tag amp w noise <<< "$spec"
  run "$tag" "$amp" "$w" "${noise:-0,0}" &
done
wait
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
echo ALL_DONE
