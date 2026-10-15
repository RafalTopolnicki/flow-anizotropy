#!/bin/bash
# CatBoost pilot on the 3D dataset: descriptors -> permeability tensor.
#
# Six single-target models per feature group, mirroring the 2D k11/k22/k12 runs
# (run_training.sh):
#
#   K_xx K_yy K_zz   the diagonal
#   k_xy k_xz k_yz   the off-diagonals -- the headline (NOTES 8.5)
#
# Note the case. The solver writes the raw tensor as K_ij and the SYMMETRISED
# off-diagonals as k_ij, so lowercase here is already the averaged one -- the
# analogue of 2D's k_off, not of 2D's k_xy. See NOTES 8.5a.
#
# Labels are still arriving, so this trains on whatever the last merge produced:
#   python scripts/solve_permeability3d.py --dataset DATA/aniso3d --merge-only
# Every run records the snapshot it used in arguments.json.
#
# Cost: the 3D descriptors are 5787 features against 1124 in 2D, so this is the
# expensive arm. Start with CB_GROUPS="fab porosity tda" to get the headline
# comparison before committing to the full grid, or LIMIT=800 for a first look.
#
# CB_GROUPS, not GROUPS: bash ignores assignments to GROUPS, which is a built-in
# array of the caller's group ids. run_training.sh names it the same way.
set -e

DATASET=${DATASET:-DATA/aniso3d}
DESC=${DESC:-DESC/aniso3d}
RESULTS=${RESULTS:-RESULTS/aniso3d}
TARGETS=${TARGETS:-"K_xx K_yy K_zz k_xy k_xz k_yz"}
CB_GROUPS=${CB_GROUPS:-"tda ecp ph baselines fab fabdir tpc por porosity"}
FOLDS=${FOLDS:-5}
SEED=${SEED:-0}
ITERATIONS=${ITERATIONS:-1500}
LIMIT=${LIMIT:-0}

EXTRA=""
if [ "$LIMIT" != "0" ]; then EXTRA="--limit $LIMIT"; fi

echo "labels   $DATASET/permeability.csv  ($(($(wc -l < "$DATASET/permeability.csv") - 1)) rows)"
echo "targets  $TARGETS"
echo "groups   $CB_GROUPS"
echo "results  $RESULTS/catboost"
echo

for T in $TARGETS; do
  echo "=== CatBoost | target $T ==="
  python scripts/train_catboost.py \
    --dataset "$DATASET" --labels "$DATASET/permeability.csv" \
    --descriptors "$DESC/descriptors.csv" "$DESC/baselines.csv" \
    --groups $CB_GROUPS --targets "$T" \
    --folds "$FOLDS" --seed "$SEED" --iterations "$ITERATIONS" $EXTRA \
    --output "$RESULTS/catboost/$T"
  echo
done

python scripts/collect_results.py --results "$RESULTS" --target k_xy
