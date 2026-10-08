#!/usr/bin/env bash
# N202b -- the amplitude lever, LEGAL range only. 0.030 is geometrically illegal (C1 max
# turn 150.6 deg, 300/300 harness errors) because the loop amplitude exceeds the row
# pitch half-width; 0.0225 is the tightest legal upper arm.  fixture_A is a ROUND face,
# so its yaw is unobservable and the post-registration residual IS the prior -- one level
# of sigma_R spans the whole curve, which is why 1 level suffices and 4 amplitudes are
# affordable.  Paired in-rig against champion amplitude 0.015 on the SAME seeds (G4).
set -u
export AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_CASTS=0 AEGIS_REG_DFACT=2.0 AEGIS_REG_CN=16 AEGIS_REG_N=32
export AEGIS_BASE_SEED=400000 AEGIS_POSE_NOISE="0,56"
for amp in 0.010 0.0225; do
  tag=$(echo "$amp" | tr '.' 'p')
  AEGIS_TROCH_AMP_M="$amp" timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
    --seeds 200 --no-upload --path trochoid --compare trochoid \
    --compare-env AEGIS_TROCH_AMP_M=0.015 --suites fixture_A \
    --out "results/aegis_v2/N202b_r312_s200_amp${tag}.jsonl" > "experiments/N202b_r312_s200_amp${tag}.log" 2>&1
  echo "amp=$amp rc=$?"
  pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
done
AEGIS_TROCH_AMP_M=0.015 timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
  --seeds 200 --no-upload --path trochoid --compare trochoid \
  --compare-env AEGIS_TROCH_AMP_M=0.015 --suites fixture_A \
  --out "results/aegis_v2/N202b_r312_s200_amp0p015.jsonl" > "experiments/N202b_r312_s200_amp0p015.log" 2>&1
echo "amp=0.015 rc=$?"
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
