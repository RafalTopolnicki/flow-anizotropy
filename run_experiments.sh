#!/bin/bash
# ---------------------------------------------------------------------------
# Every experiment, as a plain list of calls.  Read it top to bottom; run the
# whole file, or copy out any single line and run that.
#
# Every target is run BOTH WAYS over the SAME nine feature groups:
#   single  a model fitted on that target alone
#   joint   a multi-output model fitted on the whole set at once, scored on
#           each of its outputs
# so "does joint training help this component" is answerable for every target,
# not just for some of them.  (The previous version covered k_xx/k_yy on only
# five groups and never fitted the invariants singly at all, which left that
# comparison impossible exactly where it was most interesting.)
#
# Order is deliberate: the joint full tensor and the off-diagonal come first,
# because those are the two results the study turns on.  Stopping early still
# leaves you with the answer.
#
# RESULTS GO IN:
#   RESULTS/catboost/<target>__<group>/     results.csv    aggregate per target
#                                           metrics.json   R2/MAE/MSE, every fold
#                                           arguments.json how it was produced
#   RESULTS/cnn/<backbone>_<target>_<aug>/  same, plus oof_predictions.npy
#   RESULTS/summary.csv                     all of the above merged
#
#   python scripts/collect_results.py --results RESULTS      # any time, incl. mid-run
#
# Re-running a line overwrites that one directory and nothing else, so it is
# safe to interrupt and safe to redo a single experiment.
#
#   bash run_experiments.sh 2>&1 | tee run.log
#
# COST.  Measured anchor: the `tda` group (1124 features) took 144.6 s at 600
# rows x 2 folds, and CatBoost is about linear in row-folds and feature count.
# At 4991 rows x 5 folds that is ~50 min for `tda` on one target; the nine
# groups sum to ~5x the tda feature count, so ~4 h per target, and there are
# nine target-runs.  Budget ~40 h for CatBoost, ~5 h for the CNN on the GPU.
# Set ITER=500 for a first pass: the ranking BETWEEN feature groups is what the
# study turns on and it is stable well before the last R^2 decimals are.
# ---------------------------------------------------------------------------
set -u

ITER=${ITER:-1500}

D="--dataset DATA/aniso --labels DATA/aniso/permeability.csv"
F="--descriptors DESC/aniso/descriptors.csv DESC/aniso/baselines.csv"
CV="--folds 5 --seed 0 --iterations $ITER"
CNN="--epochs 30 --batch 32 --folds 5 --seed 0"
G="tda ecp ph all baselines fab tpc por porosity"

mkdir -p RESULTS

# ===========================================================================
# 1. FULL TENSOR — joint: k_xx, k_yy, k_off in one MultiRMSE model
# ===========================================================================
for g in $G; do
python scripts/train_catboost.py $D $F $CV --targets full_tensor --multioutput --groups $g --output RESULTS/catboost/full_tensor__$g
done

# ===========================================================================
# 2. OFF-DIAGONAL — single: k12 = k_off, the symmetrised (k_xy + k_yx)/2.
#    The headline.  K = K^T is exact for Stokes flow, so symmetrising halves
#    the label noise on precisely the component that needs it (NOTES 3.3).
# ===========================================================================
for g in $G; do
python scripts/train_catboost.py $D $F $CV --targets k12 --groups $g --output RESULTS/catboost/k12__$g
done

# ===========================================================================
# 3. CNN on the raw image — same targets, same folds, same metrics schema.
#    Two augmentations, and the difference is a real experiment:
#      roll      circular shift — an exact symmetry of a periodic cell, always
#                label-preserving, free regularisation.
#      roll+d4   also quarter-turns and flips, with the LABELS transformed
#                alongside the image (a quarter-turn sends k_off -> -k_off).
#    Rotating images while holding labels fixed — what every stock pipeline
#    does — would teach that k_off is orientation-independent and destroy the
#    signal this study is about (NOTES 4.5).
# ===========================================================================
for bb in convnext_tiny resnet50 densenet121; do
python scripts/train_cnn.py $D $CNN --targets full_tensor --backbone $bb --augment roll    --output RESULTS/cnn/${bb}_full_tensor_roll
python scripts/train_cnn.py $D $CNN --targets k12         --backbone $bb --augment roll    --output RESULTS/cnn/${bb}_k12_roll
python scripts/train_cnn.py $D $CNN --targets full_tensor --backbone $bb --augment roll+d4 --output RESULTS/cnn/${bb}_full_tensor_rolld4
python scripts/train_cnn.py $D $CNN --targets k12         --backbone $bb --augment roll+d4 --output RESULTS/cnn/${bb}_k12_rolld4
done

# --- everything above answers the question; everything below completes it ---

# ===========================================================================
# 4. DIAGONAL COMPONENTS — single, k11 = k_xx and k22 = k_yy.
#    Porosity-dominated and easy; the point is the contrast with k12, so they
#    get the same nine groups rather than a reduced list.
# ===========================================================================
for tgt in k11 k22; do
for g in $G; do
python scripts/train_catboost.py $D $F $CV --targets $tgt --groups $g --output RESULTS/catboost/${tgt}__$g
done
done

# ===========================================================================
# 5. INVARIANTS — joint (all four at once), then each one singly.
#    theta is defined mod 180 deg, so it is learned as (cos 2t, sin 2t) and
#    never regressed directly.
# ===========================================================================
for g in $G; do
python scripts/train_catboost.py $D $F $CV --targets invariant --multioutput --groups $g --output RESULTS/catboost/invariant__$g
done

for tgt in log_k_mean log_k_ratio cos2theta sin2theta; do
for g in $G; do
python scripts/train_catboost.py $D $F $CV --targets $tgt --groups $g --output RESULTS/catboost/${tgt}__$g
done
done

# ===========================================================================
# 6. CNN on the remaining targets, one backbone each — the CNN comparison is
#    established in section 3; these fill in the table.
# ===========================================================================
for tgt in k11 k22 invariant log_k_mean log_k_ratio cos2theta sin2theta; do
python scripts/train_cnn.py $D $CNN --targets $tgt --backbone densenet121 --augment roll --output RESULTS/cnn/densenet121_${tgt}_roll
done

# ===========================================================================
python scripts/collect_results.py --results RESULTS
