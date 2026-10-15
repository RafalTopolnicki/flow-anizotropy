#!/bin/bash
# Every descriptor for a REAL-ROCK 3D set (default DATA/rock_berea), one driver:
#
#   1. baselines   por_ tpc_ fab_ fabdir_            -> <PREFIX>_cone/baselines.csv
#   2. cone        ecp_ + ph_, nine directions       -> <PREFIX>_cone/descriptors.csv
#   3. throat      thr_, nine directions             -> <PREFIX>_throat/descriptors.csv
#   4. throat iso  thr_a1_0_0_, one pass             -> <PREFIX>_throat_iso/descriptors.csv
#
# The roots mirror the synthetic ones (DESC/aniso3d_ph_fixed, _throat,
# _throat_iso), so training is the same call with these three CSVs:
#
#   --descriptors DESC/rock_berea_cone/descriptors.csv \
#                 DESC/rock_berea_throat/descriptors.csv \
#                 DESC/rock_berea_cone/baselines.csv
#
# Needs no labels: run it while the LBM is still going. Resumable: every stage
# skips work already done (per-chunk markers), so rerun after an interruption.
#
#   SCRATCH=/local/scratch/$USER/rockdesc WORKERS=192 \
#       nohup ./run_rock_descriptors.sh > ROCK_DESC.log 2>&1 &
#
# SCRATCH must be a LOCAL disk, not /home: the intermediates are ~28k .npy
# files (/home has a 1M-inode quota that is already nearly full) and a 1000-
# structure chunk of 3-channel filtrations is ~6 GB. Only the final CSVs land
# under DESC/. Clearing SCRATCH before stage 5 finishes loses the .npy cache and
# the next run recomputes.
#
# Imaging -- measured on Berea, NOT copied (2026-10-04, 16 cubes, rows 0-15,
# directions [1,0,0] [1,1,0] [0,1,-1]; rule of NOTES 11.11: each dimension's
# range covers its own p99, bandwidth = geometric mean of the two pixel sizes):
#
#   cone       H0 pers p99   H1 pers p99   H2 pers p99   H2 birth p1-p99
#   Berea        .42-.45       .96-1.00      .17-.55       .38-.99
#   synthetic    1.03          .72           .02           .48-.98
#
#   The cone changes most. The synthetic H2 window p[0,0.06] would put nearly
#   every Berea H2 bar in its top row, so the cone gets new windows. Top-1
#   variance share of each 100-px block on the 16 cubes, synthetic -> these
#   settings: H0 .90->.82, H1 .91->.84, H2 .49->.42 ([1,0,0]); none SMEARED.
#
#   throat     H0 pers p99   H1 pers p99   H2         births
#   Berea        .28-.35       .35-.47     all at (.9, .35)   <= .9
#   -> the synthetic throat windows already cover Berea's p99 with margin, so
#      they are KEPT, which leaves thr_ columns imaged identically on both sets.
#      Iso H0 p99 .289 sat at the synthetic edge .30, so that one is widened
#      to .35. RCAP=10 caps .03% of Berea pore voxels (synthetic .85%).
#
#   Known, not an imaging bug: throat H2 (and H1 along the axes) is nearly
#   rank 1 on Berea, as H2 is on synthetic (.999) -- most bars sit on one
#   point (born at the 1-voxel level .9, die at the solid value 1.25), so the
#   block is effectively a count.
set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")"

DATASET=${DATASET:-DATA/rock_berea}
PREFIX=${PREFIX:-DESC/$(basename "$DATASET")}
if [ -z "$SCRATCH" ]; then
  echo "ABORT: set SCRATCH to a local scratch directory (not /home) -- see header."
  exit 1
fi
WORKERS=${WORKERS:-$(nproc)}
JLENV=${JLENV:-$HOME/direction-aware-tda-for-porous-materials}
CHUNK=${CHUNK:-1000}
LIMIT=${LIMIT:-0}
KEEP_FILT=${KEEP_FILT:-0}

# Never write into a synthetic root: these hold every recorded 3D number.
case "$PREFIX" in
  *aniso*) echo "ABORT: PREFIX=$PREFIX points at a synthetic descriptor root."; exit 1 ;;
esac

CONEROOT=${PREFIX}_cone
THRROOT=${PREFIX}_throat
ISOROOT=${PREFIX}_throat_iso
NPY=$SCRATCH/npy
FILTROOT=$SCRATCH/filt

N_TOTAL=$(( $(wc -l < "$DATASET/structures.csv") - 1 ))
if [ "$LIMIT" != "0" ]; then N_TOTAL=$LIMIT; fi

# --- parameters (see header) -------------------------------------------------
# Every imaging range/bandwidth below can be overridden from the environment
# (e.g. run_sst11_descriptors.sh re-images cone H2 and iso H0); the defaults are
# the Berea measurement and are what DESC/rock_berea_* were built with.
# cone filtration: as recorded for the synthetic set
RADIUS=4; CONE_R=3; CONE_H=6; GRID_RES=6
C_RANGE_H0=${C_RANGE_H0:-"0 1.0 0 0.50"};   C_BW_H0=${C_BW_H0:-0.071}
C_RANGE_H1=${C_RANGE_H1:-"0 1.0 0 1.10"};   C_BW_H1=${C_BW_H1:-0.105}
C_RANGE_H2=${C_RANGE_H2:-"0.3 1.0 0 0.60"}; C_BW_H2=${C_BW_H2:-0.065}
# throat filtration: as recorded for the synthetic set
RCAP=10; HORIZON=6; PHASE=void
T_RANGE_H0=${T_RANGE_H0:-"0 0.9 0 0.40"};   T_BW_H0=${T_BW_H0:-0.060}
T_RANGE_H1=${T_RANGE_H1:-"0 0.9 0 0.75"};   T_BW_H1=${T_BW_H1:-0.082}
T_RANGE_H2=${T_RANGE_H2:-"0 0.9 0 0.50"};   T_BW_H2=${T_BW_H2:-0.067}
I_RANGE_H0=${I_RANGE_H0:-"0 0.9 0 0.35"};   I_BW_H0=${I_BW_H0:-0.056}      # widened from .30, see header
I_RANGE_H1=${I_RANGE_H1:-"0 0.9 0 0.95"};   I_BW_H1=${I_BW_H1:-0.092}
I_RANGE_H2=${I_RANGE_H2:-"0 0.9 0 0.75"};   I_BW_H2=${I_BW_H2:-0.082}
PH_WEIGHT=linear; PH_RES="10 10"; N_POR=80; N_TPC=41

DIRECTIONS=("1 0 0" "0 1 0" "0 0 1" \
            "1 1 0" "1 -1 0" \
            "1 0 1" "1 0 -1" \
            "0 1 1" "0 1 -1")

mkdir -p "$CONEROOT" "$THRROOT" "$ISOROOT" "$NPY" "$FILTROOT"
echo "dataset $DATASET  ($N_TOTAL structures)   workers $WORKERS   scratch $SCRATCH"
echo "roots   $CONEROOT  $THRROOT  $ISOROOT"

ph () {   # ph <filtdir> <outdir> <logfile> <r0> <b0> <r1> <b1> <r2> <b2>
  python scripts/ph3d.py --inputdir "$1" --outputdir "$2" \
      --weight "$PH_WEIGHT" --resolution $PH_RES \
      --im_range_h0 $4 --bandwidth_h0 "$5" \
      --im_range_h1 $6 --bandwidth_h1 "$7" \
      --im_range_h2 $8 --bandwidth_h2 "$9" \
      --workers "$WORKERS" | tee "$3"
  if grep -q SMEARED "$3"; then
    echo "ABORT: ph3d reports a SMEARED persistence image ($3)."; exit 1
  fi
}

# --- 1. baselines --------------------------------------------------------------
BAS=$CONEROOT/baselines.csv
if [ -f "$BAS" ] && [ $(( $(wc -l < "$BAS") - 1 )) -eq "$N_TOTAL" ]; then
  echo "=== 1/4 baselines -- already complete, skipping ==="
else
  echo "=== 1/4 baselines  $(date -Is) ==="
  python scripts/baselines3d.py --dataset "$DATASET" --output "$BAS" \
      --n_por "$N_POR" --n_tpc "$N_TPC" --workers "$WORKERS" --limit "$LIMIT"
fi

# --- 2. cone: filtration (3 channels) -> ECP + PH -> drop ----------------------
for D in "${DIRECTIONS[@]}"; do
  TAG=${D// /_}
  for START in $(seq 0 "$CHUNK" $((N_TOTAL - 1))); do
    TAKE=$CHUNK; [ $((START + TAKE)) -gt "$N_TOTAL" ] && TAKE=$((N_TOTAL - START))
    MARK=$NPY/cone/.done_a${TAG}_${START}_${TAKE}
    if [ -f "$MARK" ]; then echo "=== 2/4 cone [$D] $START+$TAKE -- done, skipping ==="; continue; fi
    echo "=== 2/4 cone [$D] structures $START..$((START + TAKE))  $(date -Is) ==="
    FILT=$FILTROOT/cone_a$TAG
    python scripts/filtration3d.py --dataset "$DATASET" --outputdir "$FILT" \
        --direction $D --radius "$RADIUS" \
        --cone-radius "$CONE_R" --cone-height "$CONE_H" \
        --workers "$WORKERS" --start "$START" --limit "$TAKE"
    julia --project="$JLENV" --threads "$WORKERS" scripts/ecp3d.jl \
        "$FILT" "$NPY/cone/ecp/a$TAG" --grid-res "$GRID_RES" | tee "$CONEROOT/ecp_a$TAG.log"
    # ecp3d.jl writes zeros and carries on when one structure fails.
    if grep -q "Error in ECP" "$CONEROOT/ecp_a$TAG.log"; then
      echo "ABORT: ECP failed on some structures, see $CONEROOT/ecp_a$TAG.log"; exit 1
    fi
    ph "$FILT" "$NPY/cone/ph/a$TAG" "$CONEROOT/ph_params_a$TAG.log" \
       "$C_RANGE_H0" $C_BW_H0 "$C_RANGE_H1" $C_BW_H1 "$C_RANGE_H2" $C_BW_H2
    [ "$KEEP_FILT" = "1" ] || rm -rf "$FILT"
    touch "$MARK"
  done
done

# --- 3+4. throat, directional and iso -------------------------------------------
throat () {   # throat <root> <mode> <direction> <r0> <b0> <r1> <b1> <r2> <b2>
  local ROOT=$1 MODE=$2 D=$3 TAG=${3// /_} NAME; NAME=$(basename "$1")
  for START in $(seq 0 "$CHUNK" $((N_TOTAL - 1))); do
    local TAKE=$CHUNK; [ $((START + TAKE)) -gt "$N_TOTAL" ] && TAKE=$((N_TOTAL - START))
    local MARK=$NPY/$NAME/.done_a${TAG}_${START}_${TAKE}
    if [ -f "$MARK" ]; then echo "=== $NAME [$D] $START+$TAKE -- done, skipping ==="; continue; fi
    echo "=== $NAME mode=$MODE [$D] structures $START..$((START + TAKE))  $(date -Is) ==="
    local FILT=$FILTROOT/${NAME}_a$TAG
    python scripts/filtration_throat3d.py --dataset "$DATASET" \
        --outputdir "$FILT" --mode "$MODE" --direction $D \
        --rcap "$RCAP" --horizon "$HORIZON" --phase "$PHASE" \
        --workers "$WORKERS" --start "$START" --limit "$TAKE"
    ph "$FILT" "$NPY/$NAME/thr/a$TAG" "$ROOT/ph_params_a$TAG.log" "${@:4}"
    [ "$KEEP_FILT" = "1" ] || rm -rf "$FILT"
    touch "$MARK"
  done
}
echo "=== 3/4 throat  $(date -Is) ==="
for D in "${DIRECTIONS[@]}"; do
  throat "$THRROOT" dir "$D" "$T_RANGE_H0" $T_BW_H0 "$T_RANGE_H1" $T_BW_H1 "$T_RANGE_H2" $T_BW_H2
done
echo "=== 4/4 throat iso  $(date -Is) ==="
throat "$ISOROOT" iso "1 0 0" "$I_RANGE_H0" $I_BW_H0 "$I_RANGE_H1" $I_BW_H1 "$I_RANGE_H2" $I_BW_H2

# --- 5. collect into CSVs (fails on any missing structure or NaN) -------------------
echo "=== collect  $(date -Is) ==="
python scripts/collect_descriptors.py --dataset "$DATASET" --descroot "$NPY/cone" \
    --output "$CONEROOT/descriptors.csv" --kinds ecp ph --limit "$LIMIT"
for ROOT in "$THRROOT" "$ISOROOT"; do
  python scripts/collect_descriptors.py --dataset "$DATASET" \
      --descroot "$NPY/$(basename "$ROOT")" --output "$ROOT/descriptors.csv" \
      --kinds thr --limit "$LIMIT"
  ln -sfn "../$(basename "$CONEROOT")/baselines.csv" "$ROOT/baselines.csv"
done

for ROOT in "$CONEROOT" "$THRROOT" "$ISOROOT"; do
  cat > "$ROOT/config.txt" <<EOF
built $(date -I) by run_rock_descriptors.sh on $(hostname), $N_TOTAL structures of $DATASET
cone:   filtration3d.py radius $RADIUS cone $CONE_R/$CONE_H, ECP grid $GRID_RES
        PH H0 $C_RANGE_H0 bw $C_BW_H0 | H1 $C_RANGE_H1 bw $C_BW_H1 | H2 $C_RANGE_H2 bw $C_BW_H2
throat: filtration_throat3d.py rcap $RCAP horizon $HORIZON phase $PHASE
   dir  PH H0 $T_RANGE_H0 bw $T_BW_H0 | H1 $T_RANGE_H1 bw $T_BW_H1 | H2 $T_RANGE_H2 bw $T_BW_H2
   iso  PH H0 $I_RANGE_H0 bw $I_BW_H0 | H1 $I_RANGE_H1 bw $I_BW_H1 | H2 $I_RANGE_H2 bw $I_BW_H2
PH weight $PH_WEIGHT resolution $PH_RES. baselines: n_por $N_POR n_tpc $N_TPC.
${IMAGING_NOTE:-Imaging measured on Berea 2026-10-04, see the header of run_rock_descriptors.sh.}
EOF
done

echo "ROCK DESCRIPTORS DONE  $(date -Is)"
