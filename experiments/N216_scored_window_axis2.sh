#!/usr/bin/env bash
# N216 (run 365) PART 2 -- the arms the first pass could not settle.
#  (a) the ROUND-face u-cliff is bracketed far coarser than the elongated one (A/R fail at 0.28,
#      B survives to 0.34), so tighten the round ladder to resolve the corner-radius crossing;
#  (b) X6 was pre-registered on raster at 0,0, but I22's row-centring put raster at the 1.0000
#      CEILING, so the granularity jump is invisible there. Re-run the same side ladder on
#      raster at pose noise (0.03,6), the same non-saturated operating point the segment has
#      used for every granularity statement.
# Same knobs, same pairing, same 20 seeds, timeout 1200 on every arm (G5).
set -u
RIG=experiments/kaggle_aegis_sweep.py
OUT=results/aegis_v2
BASE="--compare-env AEGIS_PATCH_HU_M=0,AEGIS_PATCH_SIDE_M=0,AEGIS_COV_KERNEL=1"

one () {  # name hu side seeds noise path
  local name=$1 hu=$2 sd=$3 seeds=$4 noise=$5 path=$6
  AEGIS_PATCH_HU_M=$hu AEGIS_PATCH_SIDE_M=$sd AEGIS_POSE_NOISE=$noise AEGIS_COV_KERNEL=1 \
  timeout 1200 python3 $RIG --seeds $seeds --no-upload --path $path --compare trochoid \
    $BASE --suites fixture_A,fixture_B,fixture_R \
    --out $OUT/N216_r365b_${name}.jsonl > $OUT/N216_r365b_${name}.log 2>&1
  echo "  [$name] exit=$? hu=$hu side=$sd seeds=$seeds noise=$noise path=$path"
}
export -f one
export RIG OUT BASE

cat <<'PLAN' | grep -v '^#' | xargs -P 5 -n 6 bash -c 'one "$0" "$1" "$2" "$3" "$4" "$5"'
# (a) round-face corner-radius ladder -- tighten between 0.25 (1.0000) and 0.28 (0.8686)
hu_255      0.255   0       20  0,0    trochoid
hu_260      0.260   0       20  0,0    trochoid
hu_265      0.265   0       20  0,0    trochoid
hu_270      0.270   0       20  0,0    trochoid
hu_275      0.275   0       20  0,0    trochoid
# raster at the same operating points: its plan corner is INSCRIBED, so its cliff must sit later
ras_310     0.310   0       20  0,0    raster
ras_315     0.315   0       20  0,0    raster
# the elongated (fixture_B) u-cliff, tightened between 0.32 (1.0000) and 0.34 (0.7452)
hu_330      0.330   0       20  0,0    trochoid
hu_335      0.335   0       20  0,0    trochoid
ras_350     0.350   0       20  0,0    raster
# (b) X6 on a NON-saturated arm: raster at (0.03,6), the segment's standing operating point
X6_n140     0       0.140   20  0.03,6  raster
X6_n145     0       0.145   20  0.03,6  raster
X6_n149     0       0.149   20  0.03,6  raster
X6_n150     0       0.150   20  0.03,6  raster
X6_n151     0       0.151   20  0.03,6  raster
X6_n155     0       0.155   20  0.03,6  raster
X6_n160     0       0.160   20  0.03,6  raster
# the same granularity question on the trochoid at (0.03,6), for the mode contrast
X6_t149     0       0.149   20  0.03,6  trochoid
X6_t150     0       0.150   20  0.03,6  trochoid
X6_t151     0       0.151   20  0.03,6  trochoid
# control: the same arm at the FROZEN side (0.12) and (0.18) so the jump is bracketed
X6_f120     0       0.120   20  0.03,6  raster
X6_f180     0       0.180   20  0.03,6  raster
PLAN
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
echo N216B_DONE
