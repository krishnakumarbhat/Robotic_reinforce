#!/usr/bin/env bash
# N213 (run 362): POSE-NOISE-OFFSET AXIS. Held planning offsets, paired in-rig against the
# frozen champion on the SAME seeds (--compare-env zeroes both knobs for the baseline arm).
# Local CPU only, G1: no GPU, nothing downloaded. Every arm is `timeout 1200`-wrapped (G5).
# Arms are independent processes (PyBullet DIRECT, 1 thread each), so they run 5-wide.
set -u
RIG=experiments/kaggle_aegis_sweep.py
OUT=results/aegis_v2
BASE="--compare-env AEGIS_POSE_FIX_U=0,AEGIS_POSE_FIX_V=0"

one () {  # name u v seeds noise
  local name=$1 u=$2 v=$3 seeds=$4 noise=$5
  AEGIS_POSE_FIX_U=$u AEGIS_POSE_FIX_V=$v AEGIS_POSE_NOISE=$noise \
  timeout 1200 python3 $RIG --seeds $seeds --no-upload --path trochoid --compare trochoid \
    $BASE --suites fixture_A,fixture_B,fixture_R \
    --out $OUT/N213_r362_${name}.jsonl > $OUT/N213_r362_${name}.log 2>&1
  echo "  [$name] exit=$? u=$u v=$v seeds=$seeds noise=$noise"
}
export -f one
export RIG OUT BASE

# held-offset ladder: the offset IS the dose (20 seeds = the G4 minimum)
# stochastic ladder: the declared level vs the realized draw (100 seeds: the tail needs draws)
cat <<'PLAN' | xargs -P 5 -n 5 bash -c 'one "$0" "$1" "$2" "$3" "$4"'
id         0     0     20  0,0
v_p020     0     0.02  20  0,0
v_p030     0     0.03  20  0,0
v_p040     0     0.04  20  0,0
v_p050     0     0.05  20  0,0
v_p060     0     0.06  20  0,0
v_p080     0     0.08  20  0,0
v_p100     0     0.10  20  0,0
v_p120     0     0.12  20  0,0
v_n050     0    -0.05  20  0,0
v_n080     0    -0.08  20  0,0
v_n100     0    -0.10  20  0,0
v_n120     0    -0.12  20  0,0
u_p050     0.05  0     20  0,0
u_p100     0.10  0     20  0,0
u_p150     0.15  0     20  0,0
u_p200     0.20  0     20  0,0
u_n100    -0.10  0     20  0,0
u_n150    -0.15  0     20  0,0
d_100_050  0.10  0.05  20  0,0
d_150_060  0.15  0.06  20  0,0
sig_0012   0     0    100  0.01,2
sig_0306   0     0    100  0.03,6
sig_0510   0     0    100  0.05,10
PLAN
echo N213_LADDER_DONE
