#!/usr/bin/env bash
# N216 (run 365): the SCORED-WINDOW axis -- the metric's DENOMINATOR, and the last unvaried
# literal group. The scrub patch is the literal `half = 0.20`, `side = 0.12/0.18` in four places
# for all 457 logged rows, and scrub_grid floors it to [nu*CELL_M, nv*CELL_M] cells: the scored
# window is 0.40 x 0.10 m of a DECLARED 0.40 x 0.12 m patch on the elongated suites (v truncated
# 16.67%, asymmetric about the patch centre) and 0.40 x 0.15 of 0.40 x 0.18 on the round ones.
# Every coverage_cont in the segment is a FRACTION OF AN UNREPORTED WINDOW, and "transfer to a
# new fixture" has never been tested against a resized task.
# Two diagnostic knobs (AEGIS_PATCH_HU_M / AEGIS_PATCH_SIDE_M, both default 0.0 = the frozen
# literal, single source `patch_extents`) turn the patch into a controlled dose. The patch is the
# TASK: plan rows and scored window move together. `_coverage_cont` is left VERBATIM, the press
# law and the solver are untouched, and the head is never teleported, so coverage_cont and
# success still come only from physics contacts at return time (G7). The `declared` KERNEL is
# diagnostic only (same contacts, full declared patch).
# Predictions X1-X6 are in the rig source (KNOB_GLOBALS comment) and equations.md ROW N216,
# written before the first run. Every arm is paired in-rig against the frozen champion on the
# SAME 20 seeds via --compare-env (G4). Local CPU only, G1. timeout 1200 on every arm (G5) and
# the anchored kill after the batch.
set -u
RIG=experiments/kaggle_aegis_sweep.py
OUT=results/aegis_v2
BASE="--compare-env AEGIS_PATCH_HU_M=0,AEGIS_PATCH_SIDE_M=0,AEGIS_COV_KERNEL=1"

one () {  # name hu side seeds noise path facehu facehv padhu padhv padmass
  local name=$1 hu=$2 sd=$3 seeds=$4 noise=$5 path=$6 fhu=$7 fhv=$8 phu=$9 phv=${10} pm=${11}
  AEGIS_PATCH_HU_M=$hu AEGIS_PATCH_SIDE_M=$sd AEGIS_POSE_NOISE=$noise \
  AEGIS_FACE_HU_M=$fhu AEGIS_FACE_HV_M=$fhv \
  AEGIS_PAD_HU_M=$phu AEGIS_PAD_HV_M=$phv AEGIS_PAD_MASS=$pm \
  AEGIS_COV_KERNEL=1 \
  timeout 1200 python3 $RIG --seeds $seeds --no-upload --path $path --compare trochoid \
    $BASE --suites fixture_A,fixture_B,fixture_R \
    --out $OUT/N216_r365_${name}.jsonl > $OUT/N216_r365_${name}.log 2>&1
  echo "  [$name] exit=$? hu=$hu side=$sd seeds=$seeds noise=$noise path=$path face=$fhu/$fhv pad=$phu/$phv/$pm"
}
export -f one
export RIG OUT BASE

cat <<'PLAN' | grep -v '^#' | xargs -P 5 -n 11 bash -c 'one "$0" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}"'
# identity: must reproduce the frozen champion bit-for-bit on coverage_cont / success / escaped
id          0       0       20  0,0    trochoid  0     0     0     0     0
# X1: ACROSS-THE-PATCH (u) ladder. Trochoid plan reach = half + 0.023 (measured from scrub_uv),
# face half 0.34 -> cliff predicted in (0.30, 0.32]; raster reach = half exactly -> (0.34, 0.36].
hu_150      0.150   0       20  0,0    trochoid  0     0     0     0     0
hu_250      0.250   0       20  0,0    trochoid  0     0     0     0     0
hu_280      0.280   0       20  0,0    trochoid  0     0     0     0     0
hu_300      0.300   0       20  0,0    trochoid  0     0     0     0     0
hu_310      0.310   0       20  0,0    trochoid  0     0     0     0     0
hu_320      0.320   0       20  0,0    trochoid  0     0     0     0     0
hu_340      0.340   0       20  0,0    trochoid  0     0     0     0     0
hu_360      0.360   0       20  0,0    trochoid  0     0     0     0     0
# X2: the raster discriminator -- no loop excursion, so the cliff must move +0.023 m later
ras_300     0.300   0       20  0,0    raster    0     0     0     0     0
ras_320     0.320   0       20  0,0    raster    0     0     0     0     0
ras_340     0.340   0       20  0,0    raster    0     0     0     0     0
ras_360     0.360   0       20  0,0    raster    0     0     0     0     0
# X1b: the ACROSS-PATCH (v) ladder. Plan v reach = side/2 + 0.015, face half 0.14 (elongated)
# -> cliff predicted where side/2 + 0.015 > 0.14, i.e. side > 0.25.
sd_060      0       0.060   20  0,0    trochoid  0     0     0     0     0
sd_090      0       0.090   20  0,0    trochoid  0     0     0     0     0
sd_100      0       0.100   20  0,0    trochoid  0     0     0     0     0
sd_140      0       0.140   20  0,0    trochoid  0     0     0     0     0
sd_180      0       0.180   20  0,0    trochoid  0     0     0     0     0
sd_220      0       0.220   20  0,0    trochoid  0     0     0     0     0
sd_240      0       0.240   20  0,0    trochoid  0     0     0     0     0
sd_260      0       0.260   20  0,0    trochoid  0     0     0     0     0
sd_280      0       0.280   20  0,0    trochoid  0     0     0     0     0     0
# X5: is the cliff set by the pad/window RATIO? bigger pad (r_eff 0.050 for all three tools)
hu_300_pad  0.300   0       20  0,0    trochoid  0     0     0.060 0.050 0
hu_320_pad  0.320   0       20  0,0    trochoid  0     0     0.060 0.050 0
# X4: SIMILARITY arms -- patch + face + pad footprint + pad mass all scaled together, so the
# task gets bigger in every dimension at once. Coverage must stay 1.0000 on fixture_B.
# The pad is set to the MEAN frozen pad (hu 0.060, hv 0.0417, mass 0.0923 kg) times the scale
# factor, because the N214 pad knobs are absolute and one pair cannot keep three tools' aspect
# ratios; the cost is that the sim arms have no tool-to-tool footprint spread (stated).
sim_120     0.240   0.144   20  0,0    trochoid  0.408 0.168 0.072 0.050 0.111
sim_150     0.300   0.180   20  0,0    trochoid  0.510 0.210 0.090 0.063 0.138
# the responsive band (0.03,6) on the frozen task and on the two most informative enlargements
hu_300n     0.300   0       20  0.03,6 trochoid  0     0     0     0     0
hu_320n     0.320   0       20  0.03,6 trochoid  0     0     0     0     0
sim_150n    0.300   0.180   20  0.03,6 trochoid  0.510 0.210 0.090 0.063 0.138
# X6: metric granularity. nv = int(side/CELL_M) steps 2 -> 3 between 0.149 and 0.150, so the
# scored window AND the plan row count jump together; run it on raster (coverage 0.911 at 0,0,
# not saturated) so the jump is visible.
sd_149r     0       0.149   20  0,0    raster    0     0     0     0     0
sd_150r     0       0.150   20  0,0    raster    0     0     0     0     0
sd_151r     0       0.151   20  0,0    raster    0     0     0     0     0
sd_140r     0       0.140   20  0,0    raster    0     0     0     0     0
sd_190r     0       0.190   20  0,0    raster    0     0     0     0     0
PLAN
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
echo N216_LADDER_DONE
