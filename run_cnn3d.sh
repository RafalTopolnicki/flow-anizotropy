#!/bin/bash
# CNN pilot on the 3D dataset: 80^3 volume -> permeability tensor.
#
# The GPU machine. Same six single targets as run_catboost3d.sh, so the rows
# concatenate into one table:
#
#   K_xx K_yy K_zz   the diagonal
#   k_xy k_xz k_yz   the off-diagonals -- the headline (NOTES 8.5)
#
# Needs only torch (plus sklearn/pandas/numpy) -- no timm, no MONAI; the 3D
# DenseNet is written out in scripts/train_cnn3d.py. Copy over DATA/aniso3d/
# structures/ (2.6 GB), DATA/aniso3d/permeability.csv and its manifest, and
# scripts/. Check the pipeline with no GPU and no data first:
#
#   python scripts/train_cnn3d.py --self-test
#
# AUGMENT=roll is the random circular shift: exact on a periodic cell and
# label-preserving, so it works with any target. The 48-element octahedral group
# (roll+oct) needs a COMPLETE family, because a permutation can send k_xy to
# +/-k_xz -- so it cannot be used on these single-target runs. Two joint runs
# afterwards are the way to use it, and they are the interesting experiment:
#
#   python scripts/train_cnn3d.py --dataset DATA/aniso3d --backbone densenet121 \
#     --targets k_xy k_xz k_yz --augment roll+oct --epochs 30 --batch 8 \
#     --output RESULTS/aniso3d/cnn3d/densenet121_offdiag_oct
#   python scripts/train_cnn3d.py --dataset DATA/aniso3d --backbone densenet121 \
#     --targets K_xx K_yy K_zz --augment roll+oct --epochs 30 --batch 8 \
#     --output RESULTS/aniso3d/cnn3d/densenet121_diag_oct
#
# Cost: batch is set by activations, not weights. 8 at 80^3 is the default;
# halve it if the card is small, and raise EPOCHS only after a first fold looks
# sane. FOLDS=1 with --only-fold is not wired in here -- use LIMIT for a smoke
# run instead.
set -e

DATASET=${DATASET:-DATA/aniso3d}
RESULTS=${RESULTS:-RESULTS/aniso3d}
TARGETS=${TARGETS:-"K_xx K_yy K_zz k_xy k_xz k_yz"}
BACKBONE=${BACKBONE:-densenet121}
AUGMENT=${AUGMENT:-roll}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-8}
FOLDS=${FOLDS:-5}
SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda}
LIMIT=${LIMIT:-0}

EXTRA=""
if [ "$LIMIT" != "0" ]; then EXTRA="--limit $LIMIT"; fi

echo "labels   $DATASET/permeability.csv  ($(($(wc -l < "$DATASET/permeability.csv") - 1)) rows)"
echo "targets  $TARGETS"
echo "backbone $BACKBONE   augment $AUGMENT   epochs $EPOCHS   batch $BATCH"
echo "results  $RESULTS/cnn3d"
echo

for T in $TARGETS; do
  echo "=== CNN $BACKBONE | target $T ==="
  python scripts/train_cnn3d.py \
    --dataset "$DATASET" --labels "$DATASET/permeability.csv" \
    --backbone "$BACKBONE" --targets "$T" \
    --augment "$AUGMENT" --epochs "$EPOCHS" --batch "$BATCH" \
    --folds "$FOLDS" --seed "$SEED" --device "$DEVICE" $EXTRA \
    --output "$RESULTS/cnn3d/${BACKBONE}_$T"
  echo
done

python scripts/collect_results.py --results "$RESULTS" --target k_xy
