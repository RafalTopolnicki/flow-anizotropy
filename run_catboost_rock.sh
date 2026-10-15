#!/bin/bash
# CatBoost on the real-rock 3D set (stretched Berea, DATA/rock_berea, n=1000).
#
# GROUPS -- the baseline families separately AND concatenated (standing
# instruction: the union alone hides which family carries the signal):
#
#   por tpc fab fabdir     the four baseline families, each on its own
#   baselines              their concatenation (por_ + tpc_ + fab_ + fabdir_)
#   tda                    ecp_ + ph_, the cone descriptors (9 directions)
#   thr                    throat-scale PH, directional (9 directions)
#   all                    baselines + tda
#   baselines+thr          baselines + thr
#
# TARGETS K_xx k_xy k_xz k_yz, one SINGLE-target fit per component, so the rows
# pair with RESULTS/rock_berea/cnn3d/densenet121_<T> (single, roll): same
# KFold(5, shuffle, seed 0), same sample_id order -> same structures per fold.
#
# Descriptors are the run_rock_descriptors.sh roots. The throat_iso CSV is NOT
# passed: its columns are named thr_a1_0_0_*, the same as the directional
# [1,0,0] block, and train_catboost silently drops duplicate columns from the
# later file -- passing both would mean `thr` is not what it says.
#
# One invocation per (target, group): train_catboost.py writes only after a
# call finishes, so this is resumable -- a run whose metrics.json exists is
# skipped. Rerun after an interruption.
#
#   nohup ./run_catboost_rock.sh > CB_ROCK.log 2>&1 &
#
# Overrides: TARGETS="k_xy" FEATGROUPS="tpc thr" OUT=... ITER=... ./run_catboost_rock.sh
set -u
cd "$(dirname "$(readlink -f "$0")")"

ITER=${ITER:-1500}
FOLDS=${FOLDS:-5}
SEED=${SEED:-0}
DATASET=${DATASET:-DATA/rock_berea}
OUT=${OUT:-RESULTS/rock_berea/catboost}
EXPECT=${EXPECT:-1000}

# NOT named GROUPS: bash reserves that name and silently discards assignments.
FEATGROUPS=${FEATGROUPS:-"por tpc fab fabdir baselines tda thr all baselines+thr"}
TARGETS=${TARGETS:-"K_xx k_xy k_xz k_yz"}

CONE=DESC/rock_berea_cone/descriptors.csv      # ecp_ + ph_
THR=DESC/rock_berea_throat/descriptors.csv     # thr_
BAS=DESC/rock_berea_cone/baselines.csv         # por_ tpc_ fab_ fabdir_

# A short CSV (a build that died partway) would inner-join down to its rows
# and hand back meaningless R^2 that looks like a result. Check.
for f in "$CONE" "$THR" "$BAS"; do
  if [ ! -f "$f" ]; then echo "ABORT: $f does not exist."; exit 1; fi
  n=$(( $(wc -l < "$f") - 1 ))
  if [ "$n" -ne "$EXPECT" ]; then
    echo "ABORT: $f has $n rows, expected $EXPECT."; exit 1
  fi
  echo "ok: $f  $n rows"
done

D="--dataset $DATASET --labels $DATASET/permeability.csv"
F="--descriptors $CONE $THR $BAS"
CV="--folds $FOLDS --seed $SEED --iterations $ITER"

mkdir -p "$OUT"
echo "groups   $FEATGROUPS"
echo "targets  $TARGETS   folds $FOLDS seed $SEED iters $ITER"
echo "output   $OUT"

for t in $TARGETS; do
  for g in $FEATGROUPS; do
    dir="$OUT/${t}__$g"
    if [ -f "$dir/metrics.json" ]; then
      echo "########## $t -- $g -- already done, skipping ##########"
      continue
    fi
    echo "########## $t -- $g  $(date -Is) ##########"
    python scripts/train_catboost.py $D $F $CV \
        --targets "$t" --groups "$g" --output "$dir"
  done
done

echo "CB ROCK DONE  $(date -Is)"
