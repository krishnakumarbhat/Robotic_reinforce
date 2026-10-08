#!/usr/bin/env bash
# N202c -- complete the amplitude dose ladder around the two knees.
#   DOWN  : A = 0.0075 (between the saturated 0.010 and the first loss 0.005) locates
#           the coverage-saturation knee A_sat on a ROUND face (fixture_A, residual yaw
#           = prior = 56 deg for every episode, so one noise level spans the whole curve).
#   UP    : A = 0.018 and A = 0.021 probe the C1/cusp knee from below; the rig ASSERTS
#           max turn < 60 deg, so an illegal arm is a measured AssertionError, not a
#           silent pass, and its log line carries the max turn in degrees.
# Env-only (AEGIS_TROCH_AMP_M is an existing R29 knob); paired in-rig against the
# champion amplitude 0.015 on the SAME seeds (G4).  Fresh seed base 400000.
set -u
export AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_CASTS=0 AEGIS_REG_DFACT=2.0
export AEGIS_REG_CN=16 AEGIS_REG_N=32
export AEGIS_BASE_SEED=400000
run () {  # run <amp> <seeds> <suites> <stem>
  local amp="$1" seeds="$2" suites="$3" stem="$4"
  AEGIS_POSE_NOISE="0,56" AEGIS_TROCH_AMP_M="$amp" timeout 1200 python3 \
    experiments/kaggle_aegis_sweep.py --seeds "$seeds" --no-upload --path trochoid \
    --compare trochoid --compare-env AEGIS_TROCH_AMP_M=0.015 --suites "$suites" \
    --out "results/aegis_v2/${stem}.jsonl" > "experiments/${stem}.log" 2>&1
  echo "amp=$amp seeds=$seeds suites=$suites rc=$?"
  pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
}
run 0.0075 200 fixture_A          N202c_r312_s200_amp0p0075
run 0.018   100 fixture_A,fixture_B,fixture_R N202c_r312_s100_amp0p018
run 0.021   100 fixture_A,fixture_B,fixture_R N202c_r312_s100_amp0p021
echo DONE
