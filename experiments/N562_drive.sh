#!/usr/bin/env bash
# N562 -- WHAT IS THE ELONGATED-FACE REGISTRATION RESIDUAL? (attribution) + DOES IT COST
# ANYTHING? (frontier probe).  Pre-registered in autoresearch_research.ideas.md (N562 block)
# BEFORE any dose.  9 paired invocations, 20 seeds x {A,B,R} x 2 arms = 2160 physical episodes.
#
# Candidate arm  = the certified N478 stack VERBATIM (no mechanism byte differs between cells
#                   except where the pre-registration names a knob: REG_N for D3, POSE_FIX for D4).
# Baseline  arm  = --compare trochoid --compare-env <frozen single-cast default>, same 20 seeds (G4).
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=results/aegis_v2
mkdir -p "$OUT"

STACK="AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0"
BASEENV="AEGIS_REG_CASTS=1.0,AEGIS_REG_CN=0.0,AEGIS_REG_DFACT=1.5,AEGIS_REG_N=32.0"

cell () {  # $1=tag  $2=pose_noise  $3=extra_candidate_env
  local tag="$1" pn="$2" extra="${3:-}"
  echo "=== N562 $tag  pose_noise=$pn  extra=${extra:-none} ==="
  timeout 1200 env AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=32 \
      AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 \
      AEGIS_POSE_NOISE="$pn" $extra \
      python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload \
        --path trochoid --compare trochoid --compare-env "$BASEENV" \
        --suites fixture_A,fixture_B,fixture_R \
        --out "$OUT/N562_${tag}.jsonl" > "$OUT/N562_${tag}.log" 2>&1
  echo "  exit=$? -> $OUT/N562_${tag}.jsonl ($(stat -c%s "$OUT/N562_${tag}.jsonl" 2>/dev/null) bytes)"
  pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
}

# ---- Phase A: attribution -------------------------------------------------
cell A1_000      "0,0"
cell A2_03264    "0.32,64"
cell A3_096_n32  "0.96,192"
cell A4_096_n128 "0.96,192"     "AEGIS_REG_N=128"
cell A5_096_fix0 "0.96,192"     "AEGIS_POSE_FIX_U=0 AEGIS_POSE_FIX_V=0"
# ---- Phase B: cost probe ABOVE the certified 1.60,320 wall ------------------
cell B1_160320   "1.60,320"
cell B2_200400   "2.00,400"
cell B3_240480   "2.40,480"
cell B4_320640   "3.20,640"

echo "N562 ALL DONE"