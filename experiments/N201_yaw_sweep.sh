#!/usr/bin/env bash
# N201 -- yaw/translation DECOUPLING on the canonical rig. Every pose-noise level in the
# programme to date ties sigma_t = sigma_R/200, so the two SE(3) terms have never been
# separated. This sweeps sigma_R at sigma_t = 0 exactly. Env-only; no rig byte is touched.
set -u
export AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_CASTS=0 AEGIS_REG_DFACT=2.0 AEGIS_REG_CN=16 AEGIS_REG_N=32
for lvl in "0,0" "0,8" "0,14" "0,18" "0,28" "0,56" "0,112"; do
  tag=$(echo "$lvl" | tr ',' '_')
  AEGIS_POSE_NOISE="$lvl" timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
    --seeds 20 --no-upload --path trochoid --compare trochoid \
    --compare-env AEGIS_REG_CASTS=1 --suites fixture_A,fixture_B,fixture_R \
    --out "results/aegis_v2/N201_r311_s20_${tag}.jsonl" > "experiments/N201_r311_s20_${tag}.log" 2>&1
  echo "level=$lvl rc=$?"
  pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
done
