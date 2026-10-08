#!/usr/bin/env bash
# N215 (run 364): the FIXTURE-FACE axis. `_build_fixture` hard-codes the two top faces
# (elongated box halfExtents [0.34, 0.14, 0.12], round cylinder radius 0.32) and no run 1-363 has
# ever moved them, yet "zero-shot transfer to a NEW fixture" is a claim about the face -- and N213
# already measured the face boundary as the exact place a plan-frame offset launches the head.
# Three diagnostic knobs (AEGIS_FACE_HU_M / AEGIS_FACE_HV_M / AEGIS_FACE_R_M, all default 0.0 =
# the frozen literal) turn the face into a controlled dose. The face HEIGHT, the scrub patch, the
# plan, the press law and the scoring kernel are untouched, so coverage_cont and success still come
# only from physics contacts at return time and nothing is teleported (G7).
# Every arm is paired in-rig against the frozen champion on the SAME 20 seeds via --compare-env
# (G4), so friction / tool / customer are identical in both halves. Local CPU only, G1.
# Every arm is `timeout 1200`-wrapped (G5); the anchored kill runs after each arm.
set -u
RIG=experiments/kaggle_aegis_sweep.py
OUT=results/aegis_v2
BASE="--compare-env AEGIS_FACE_HU_M=0,AEGIS_FACE_HV_M=0,AEGIS_FACE_R_M=0,AEGIS_COV_KERNEL=1"

one () {  # name hu hv r seeds noise path
  local name=$1 hu=$2 hv=$3 r=$4 seeds=$5 noise=$6 path=$7
  AEGIS_FACE_HU_M=$hu AEGIS_FACE_HV_M=$hv AEGIS_FACE_R_M=$r AEGIS_POSE_NOISE=$noise \
  timeout 1200 python3 $RIG --seeds $seeds --no-upload --path $path --compare trochoid \
    $BASE --suites fixture_A,fixture_B,fixture_R \
    --out $OUT/N215_r364_${name}.jsonl > $OUT/N215_r364_${name}.log 2>&1
  echo "  [$name] exit=$? hu=$hu hv=$hv r=$r seeds=$seeds noise=$noise path=$path"
}
export -f one
export RIG OUT BASE

cat <<'PLAN' | grep -v '^#' | xargs -P 5 -n 7 bash -c 'one "$0" "$1" "$2" "$3" "$4" "$5" "$6"'
# identity: must reproduce the frozen champion bit-for-bit on coverage_cont / success / escaped
id          0       0       0     20  0,0    trochoid
# F1/F2: ACROSS-PATCH (v) face ladder. Frozen 0.14, patch_v = 0.06, r_eff = 0.035/0.040/0.050.
fv_120      0       0.120   0     20  0,0    trochoid
fv_100      0       0.100   0     20  0,0    trochoid
fv_080      0       0.080   0     20  0,0    trochoid
fv_070      0       0.070   0     20  0,0    trochoid
fv_065      0       0.065   0     20  0,0    trochoid
fv_060      0       0.060   0     20  0,0    trochoid
fv_055      0       0.055   0     20  0,0    trochoid
fv_050      0       0.050   0     20  0,0    trochoid
fv_045      0       0.045   0     20  0,0    trochoid
fv_040      0       0.040   0     20  0,0    trochoid
fv_030      0       0.030   0     20  0,0    trochoid
# F1: the SAME ladder along the patch (u) axis, for the 1.27x vs 1.36x margin comparison
fu_300      0.300   0       0     20  0,0    trochoid
fu_280      0.280   0       0     20  0,0    trochoid
fu_260      0.260   0       0     20  0,0    trochoid
fu_240      0.240   0       0     20  0,0    trochoid
fu_220      0.220   0       0     20  0,0    trochoid
fu_200      0.200   0       0     20  0,0    trochoid
# F1: the ROUND face (radius; applies to fixture_A and the round half of fixture_R)
fr_300      0       0       0.300 20  0,0    trochoid
fr_280      0       0       0.280 20  0,0    trochoid
fr_260      0       0       0.260 20  0,0    trochoid
fr_240      0       0       0.240 20  0,0    trochoid
fr_220      0       0       0.220 20  0,0    trochoid
fr_200      0       0       0.200 20  0,0    trochoid
# F3: is the channel the CONTACT SET or the escape test? floor-plane fall = z_excursion
fv_020      0       0.020   0     20  0,0    trochoid
fv_010      0       0.010   0     20  0,0    trochoid
# F4: raster at the same v doses -- does the path mode move the face cliff?
ras_fv060   0       0.060   0     20  0,0    raster
ras_fv050   0       0.050   0     20  0,0    raster
ras_fv040   0       0.040   0     20  0,0    raster
# the responsive band (0.03,6) at and just below the cliff
tro_fv060n  0       0.060   0     20  0.03,6 trochoid
tro_fv055n  0       0.055   0     20  0.03,6 trochoid
PLAN
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
echo N215_LADDER_DONE
