#!/bin/bash
# Directional TDA descriptors for a generated dataset.
#
# Per direction: filtration -> ECP (Julia) + PH (gudhi) -> drop the filtration.
# The filtration is ~1 ms/structure so it is cheaper to recompute than to keep;
# holding all four directions would be ~10 GB of intermediates.
set -e

DATASET=${DATASET:-DATA/aniso}
DESCROOT=${DESCROOT:-DESC/aniso}
DIRECTIONS=${DIRECTIONS:-"0 45 90 135"}
WORKERS=${WORKERS:-$(nproc)}
GRID_RES=${GRID_RES:-8}
# gudhi defaults to a constant weight, which measured out as rank-3 images and
# 91% TPC-predictable (NOTES 11.5). DESC/aniso/ph was built that way.
PH_WEIGHT=${PH_WEIGHT:-linear}
# gudhi's default bandwidth is 1.0, which at --resolution 10 over [0,1.25] is
# EIGHT pixels and flattens the persistence image to rank 1 (NOTES 11.5a). That
# bug was fixed in ph2d.py but this driver never passed the flag, so every
# dataset built here silently got the broken representation again -- it cost a
# retracted result on 2026-09-17 (NOTES 11.11). Passed explicitly now.
# Bar statistics are NOT universal: they move with the phase and with the
# structure family (NOTES 11.8), so re-measure and re-set these for a new
# family rather than assuming these values transfer.
PH_RES=${PH_RES:-"20 20"}
PH_BW_H0=${PH_BW_H0:-0.125}
PH_BW_H1=${PH_BW_H1:-0.125}
PH_RANGE_H0=${PH_RANGE_H0:-"0 1.25 0 1.25"}
PH_RANGE_H1=${PH_RANGE_H1:-"0 1.0 0 0.10"}
RADIUS=${RADIUS:-4}
WEDGE_R=${WEDGE_R:-3}
WEDGE_H=${WEDGE_H:-6}
JLENV=${JLENV:-$HOME/direction-aware-tda-for-porous-materials}
KEEP_FILT=${KEEP_FILT:-0}

for TH in $DIRECTIONS; do
  echo "=== direction ${TH} deg ==="
  FILT=$DESCROOT/filt/a$TH
  python scripts/filtration2d.py --dataset "$DATASET" --outputdir "$FILT" \
      --direction "$TH" --radius "$RADIUS" \
      --wedge-radius "$WEDGE_R" --wedge-height "$WEDGE_H" --workers "$WORKERS"

  julia --project="$JLENV" --threads "$WORKERS" scripts/ecp2d.jl \
      "$FILT" "$DESCROOT/ecp/a$TH" --grid-res "$GRID_RES"

  python scripts/ph2d.py --inputdir "$FILT" --outputdir "$DESCROOT/ph/a$TH" \
      --weight "$PH_WEIGHT" --resolution $PH_RES \
      --bandwidth_h0 "$PH_BW_H0" --bandwidth_h1 "$PH_BW_H1" \
      --im_range_h0 $PH_RANGE_H0 --im_range_h1 $PH_RANGE_H1 \
      --workers "$WORKERS" | tee "$DESCROOT/ph_params_a$TH.log"
  if grep -q SMEARED "$DESCROOT/ph_params_a$TH.log"; then
      echo "ABORT: ph2d reports a SMEARED persistence image -- the bandwidth is"
      echo "wrong for this family. Re-measure the bar statistics and set"
      echo "PH_BW_H0/PH_BW_H1/PH_RANGE_H0/PH_RANGE_H1. See NOTES 11.8, 11.11."
      exit 1
  fi

  if [ "$KEEP_FILT" != "1" ]; then rm -rf "$FILT"; fi
done

python scripts/collect_descriptors.py --dataset "$DATASET" \
    --descroot "$DESCROOT" --output "$DESCROOT/descriptors.csv"
