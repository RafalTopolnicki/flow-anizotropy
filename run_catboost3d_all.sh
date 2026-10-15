#!/bin/bash
# 3D CatBoost: the full descriptor comparison on the complete 5000-structure
# dataset, all six tensor components.
#
# GROUPS -- the baseline families separately AND concatenated, because a
# combined `baselines` row alone hides which family carries the signal (at
# n=2816 on k_xy, `tpc` alone scored .7437 against the 1819-feature union's
# .7653, while `por` scored -.0014). Standing instruction from the user
# 2026-09-22.
#
#   por tpc fab fabdir     the four baseline families, each on its own
#   baselines              their concatenation (por_ + tpc_ + fab_ + fabdir_)
#   tda                    ecp_ + ph_, the cone descriptors
#   thr                    throat-scale PH (distance-transform filtration)
#   baselines+thr          baselines + throat
#   all                    tda + baselines
#
# DESCRIPTOR ROOTS. `tda`/`all` read DESC/aniso3d_ph_fixed, the RE-IMAGED cone
# -- NOT DESC/aniso3d, whose ph_ columns carry the rank-1 bandwidth bug (NOTES
# 12.2). Both files define ecp_ and ph_, so passing both would silently take
# whichever came first: train_catboost drops duplicate columns from the LATER
# file. Only ph_fixed is listed, so there is nothing to collide.
#
# COMPARABILITY WITH THE CNN. train_cnn3d.py and train_catboost.py both use
# `KFold(folds, shuffle=True, random_state=seed)`, both filter on conv_f*, and
# the descriptor merges preserve permeability.csv's sample_id order -- verified
# 2026-09-22 by recomputing the CNN's per-fold R^2 from its oof_predictions.npy
# under KFold(5, shuffle=True, random_state=0), which reproduced all five folds
# to five decimals. So FOLDS=5 SEED=0 scores the same structures in the same
# folds as the CNN runs in RESULTS/aniso3d/cnn3d/.
#
#   ONE CAVEAT, and it is not fixable by a flag: the CNN grid is SINGLE-target
#   (one model per component, `--targets K_xx` etc). The joint pass here fits
#   all six at once, which is a different estimator. In 2D the joint fit cost
#   the off-diagonal .024-.050 R^2 versus a single fit, and --scale-targets
#   recovered most but not all of it. For a strictly like-for-like comparison
#   with the CNN, run PASSES=single (or "joint single"); the joint pass is the
#   right one for comparing feature GROUPS against each other.
#
# SCALING. MultiRMSE minimises the summed squared error on raw scales, so the
# large-variance components dominate and the off-diagonals come out
# under-fitted. --scale-targets z-scores each target on the TRAIN fold only and
# inverts on the predictions, so every metric stays in raw units. Joint only --
# a single-target fit has nothing to balance, and the flag no-ops there.
#
# COST, extrapolated from the 2D joint pass at n=4991 and the n=2816 3D runs:
#   joint  ~5 h total   (all ~90 min, tda ~70, baselines+thr ~55, thr ~35,
#                        baselines ~25, por/fabdir ~10, tpc ~6, fab ~2)
#   single ~2x that, since each of the six components is fitted on its own.
#
#   bash run_catboost3d_all.sh 2>&1 | tee CB3D_ALL.log
#
# One invocation PER GROUP on purpose: train_catboost.py writes its output only
# after every group in a call has finished, so a single call carrying nine
# groups loses everything if it dies partway -- which is exactly how the 3D
# `all+thr` run was lost on 2026-09-18. Per-group calls make this resumable and
# let you redo one group by copying out its line.
# ---------------------------------------------------------------------------
set -u

ITER=${ITER:-1500}
FOLDS=${FOLDS:-5}
SEED=${SEED:-0}
OUT=${OUT:-RESULTS/aniso3d_all}
DATASET=${DATASET:-DATA/aniso3d}

# NOT named GROUPS: bash reserves that name for the caller's Unix group IDs and
# silently discards assignments to it. Cost one false start on 2026-09-18.
FEATGROUPS=${FEATGROUPS:-"por tpc fab fabdir baselines tda thr baselines+thr all"}

# "joint", "single", or both. joint = one MultiRMSE fit over all six components
# with --scale-targets; single = one fit per component, which is what matches
# the CNN grid.
PASSES=${PASSES:-"joint"}
TARGETS=${TARGETS:-"tensor3d"}

FIX=DESC/aniso3d_ph_fixed/descriptors.csv     # re-imaged cone: ecp_ + ph_
THR=DESC/aniso3d_throat/descriptors.csv       # throat PH: thr_
BAS=DESC/aniso3d/baselines.csv                # por_ fabdir_ tpc_ fab_

EXPECT=${EXPECT:-5000}

# A descriptor build that died partway leaves a short descriptors.csv behind;
# train_catboost would inner-join down to it, print the reduced row count and
# hand back meaningless R^2 -- a failure that looks like a result. Check.
for f in "$FIX" "$THR" "$BAS"; do
  if [ ! -f "$f" ]; then echo "ABORT: $f does not exist."; exit 1; fi
  n=$(( $(wc -l < "$f") - 1 ))
  if [ "$n" -ne "$EXPECT" ]; then
    echo "ABORT: $f has $n rows, expected $EXPECT."
    echo "Rebuild it (FROM=<its row count> ./run3d_all.sh) before training."
    exit 1
  fi
  echo "ok: $f  $n rows"
done

D="--dataset $DATASET --labels $DATASET/permeability.csv"
F="--descriptors $FIX $THR $BAS"
CV="--folds $FOLDS --seed $SEED --iterations $ITER"

mkdir -p "$OUT"
echo "groups   $FEATGROUPS"
echo "targets  $TARGETS   passes $PASSES   folds $FOLDS seed $SEED iters $ITER"
echo "output   $OUT"

case " $PASSES " in *" joint "*)
for g in $FEATGROUPS; do
  dir="$OUT/tensor_joint__$g"
  if [ -f "$dir/metrics.json" ]; then
    echo "########## joint -- $g -- already done, skipping ##########"
    continue
  fi
  echo "########## tensor3d (joint, scaled) -- $g  $(date -Is) ##########"
  python scripts/train_catboost.py $D $F $CV \
      --targets $TARGETS --multioutput --scale-targets \
      --groups "$g" --output "$dir"
done
;; esac

# Single-target pass: one fit per component, at full strength, and the pass
# that is directly comparable with RESULTS/aniso3d/cnn3d/ (which is one model
# per component).
case " $PASSES " in *" single "*)
for t in K_xx K_yy K_zz k_xy k_xz k_yz; do
  for g in $FEATGROUPS; do
    dir="$OUT/${t}__$g"
    if [ -f "$dir/metrics.json" ]; then
      echo "########## single $t -- $g -- already done, skipping ##########"
      continue
    fi
    echo "########## $t (single) -- $g  $(date -Is) ##########"
    python scripts/train_catboost.py $D $F $CV \
        --targets "$t" --groups "$g" --output "$dir"
  done
done
;; esac

python scripts/collect_results.py --results RESULTS --target k_xy
echo "CB3D ALL DONE  $(date -Is)"
