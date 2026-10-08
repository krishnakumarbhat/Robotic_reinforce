#!/usr/bin/env bash
# N214 (run 363): the TOOL-BODY FOOTPRINT axis. `tool_id = seed % 3` is the only way the head has
# ever changed in runs 1-362, and it moves the footprint (r_eff 0.035/0.040/0.050), the mass
# (0.080/0.105/0.092) and the thickness (0.012/0.030/0.006) at once, aliased with the seed. Three
# diagnostic knobs (AEGIS_PAD_HU_M / AEGIS_PAD_HV_M / AEGIS_PAD_MASS, all default 0.0 = the frozen
# per-tool value) turn the head into a controlled dose. Every arm is paired in-rig against the
# frozen champion on the SAME 20 seeds via --compare-env (G4), so friction / tool / customer are
# identical in both halves. Local CPU only, G1: no GPU, nothing downloaded. Every arm is
# `timeout 1200`-wrapped (G5) and the anchored kill runs after the ladder.
set -u
RIG=experiments/kaggle_aegis_sweep.py
OUT=results/aegis_v2
BASE="--compare-env AEGIS_PAD_HU_M=0,AEGIS_PAD_HV_M=0,AEGIS_PAD_MASS=0,AEGIS_COV_KERNEL=1"

one () {  # name hu hv mass seeds noise path
  local name=$1 hu=$2 hv=$3 mass=$4 seeds=$5 noise=$6 path=$7
  AEGIS_PAD_HU_M=$hu AEGIS_PAD_HV_M=$hv AEGIS_PAD_MASS=$mass AEGIS_POSE_NOISE=$noise \
  timeout 1200 python3 $RIG --seeds $seeds --no-upload --path $path --compare trochoid \
    $BASE --suites fixture_A,fixture_B,fixture_R \
    --out $OUT/N214_r363_${name}.jsonl > $OUT/N214_r363_${name}.log 2>&1
  echo "  [$name] exit=$? hu=$hu hv=$hv m=$mass seeds=$seeds noise=$noise path=$path"
}
export -f one
export RIG OUT BASE

cat <<'PLAN' | grep -v '^#' | xargs -P 5 -n 7 bash -c 'one "$0" "$1" "$2" "$3" "$4" "$5" "$6"'
# frozen pad, kernel ON: must reproduce the identity arm's scored pair exactly
fk          0       0       0     20  0,0    trochoid
# Q1/Q2: SQUARE ladder on r_eff (the frozen r_eff 0.035 appears as a square cell)
sq_070      0.070   0.070   0     20  0,0    trochoid
sq_050      0.050   0.050   0     20  0,0    trochoid
sq_035      0.035   0.035   0     20  0,0    trochoid
sq_025      0.025   0.025   0     20  0,0    trochoid
sq_0175     0.0175  0.0175  0     20  0,0    trochoid
sq_010      0.010   0.010   0     20  0,0    trochoid
# Q2: elongation at the FROZEN r_eff 0.035, along u (0.09x0.035) and across v (0.035x0.09)
u_long      0.090   0.035   0     20  0,0    trochoid
v_long      0.035   0.090   0     20  0,0    trochoid
# Q4: mass at the frozen footprint
m_003       0       0       0.030 20  0,0    trochoid
m_015       0       0       0.150 20  0,0    trochoid
m_030       0       0       0.300 20  0,0    trochoid
# Q5: raster at the pads where its 0.05 m row pitch starts to exceed the footprint
ras_sq025   0.025   0.025   0     20  0,0    raster
ras_sq0175  0.0175  0.0175  0     20  0,0    raster
# the responsive band (0.03,6) at the pad edge, both modes
tro_sq0175n 0.0175  0.0175  0     20  0.03,6 trochoid
ras_sq025n  0.025   0.025   0     20  0.03,6 raster
PLAN
echo N214_LADDER_DONE
