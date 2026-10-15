#!/bin/bash
# ---------------------------------------------------------------------------
# The 2D arm, recomputed.  CatBoost only -- the CNN is NOT a baseline and is
# not run here.
#
# Two passes over the same five feature groups:
#   1. the full tensor, fitted JOINTLY -- one MultiRMSE model over k_xx, k_yy
#      and k_off at once
#   2. k_off alone, because the joint fit under-weights it (see below)
#
# Five feature groups, one python call each:
#
#   baselines       por_ tpc_ fab_                       779 f
#   tda             ecp_ ph_          (cone/wedge)     3524 f
#   thr             thr_             (throat/EDT)      3200 f
#   all             ecp_ ph_ + baselines  = TDA+base   4303 f
#   baselines+thr   thr_ + baselines      = THR+base   3979 f
#   all+thr         everything            = base+TDA+THR  7503 f
#
# `all` is train_catboost.py's existing name for exactly TDA+baselines; it is
# kept rather than renamed so these rows stay comparable with every number
# already recorded under that label.
#
# TWO CHOICES WORTH KNOWING ABOUT, both one-line switches:
#
# 1. k_off, not k_xy.  In 2D the labels carry k_xy AND k_yx separately plus
#    k_off = (k_xy + k_yx)/2.  K = K^T is exact for Stokes flow, so
#    symmetrising halves the label noise on the one component that needs it
#    (NOTES 3.3), and every recorded 2D number in this study is on k_off.
#    For the raw k_xy instead: --targets k_xx k_yy k_xy.
#
# 2. TDA = DESC/aniso_ph_bw125, not DESC/aniso.  Same ECP, but the repaired
#    persistence images: the gudhi default bandwidth had flattened PH to rank 1
#    and cost .097 on k_off (NOTES 11.5c).  The original 800-feature PH loses
#    partly by construction, so scoring it would not be a fair test of TDA.
#    For the original: swap in DESC/aniso/descriptors.csv.
#
# COST.  Anchor from NOTES 11.5c: 3200 features, 4991 rows, 5 folds, single
# target = 454 s.  Joint MultiRMSE over 3 targets runs ~3x that, so roughly
#   baselines ~5 min | thr ~25 min | tda ~27 min | baselines+thr ~30 min |
#   all ~32 min   ->  ~2 h for the joint pass, ~40 min for the single k_off
#   pass, ~2.7 h total.  Set ITER=500 for a ~3x faster first pass;
# the ranking between groups settles long before the last R^2 decimals.
#
#   bash run_2d_all.sh 2>&1 | tee RUN2D.log
#
# Each group writes its own directory, so this is safe to interrupt and safe
# to redo one group by copying out its line.
# ---------------------------------------------------------------------------
set -u

ITER=${ITER:-1500}
OUT=${OUT:-RESULTS/aniso2d}
# NOT named GROUPS: bash reserves that name for the caller's Unix group IDs and
# silently DISCARDS assignments to it, so `GROUPS=... bash run_2d_all.sh` runs
# --groups <your gid> and CatBoost reports "no columns, skipping". Cost one
# false start on 2026-09-18.
# The three baseline families are fitted SEPARATELY as well as concatenated:
# `baselines` is por_(512) + tpc_(260) + fab_(7), and a combined row alone
# hides which of the three carries the signal. Standing instruction from
# the user 2026-09-22 -- always report the components alongside the union.
FEATGROUPS=${FEATGROUPS:-"por tpc fab baselines tda thr all baselines+thr all+thr"}
# Which passes to run: "joint", "single", or both.  The scaled run only wants
# the joint pass -- scaling is meaningless for a single-target fit, so a scaled
# `single` pass would just be an unscaled one under a misleading directory name.
PASSES=${PASSES:-"joint single"}

# SCALE=1 z-scores each target on the TRAIN fold before the joint fit and
# inverts it on the predictions, so MultiRMSE weighs the three components
# equally instead of by their raw variance (k_xx carries 17.8x k_off's).
# Affects the joint pass only; the single-k_off pass has nothing to balance.
SCALE=${SCALE:-}
SCALE_ARG=""
[ -n "$SCALE" ] && SCALE_ARG="--scale-targets"

TDA=DESC/aniso_ph_bw125/descriptors.csv
THR=DESC/aniso_throat/descriptors.csv
BAS=DESC/aniso/baselines.csv

D="--dataset DATA/aniso --labels DATA/aniso/permeability.csv"
F="--descriptors $TDA $THR $BAS"
CV="--folds 5 --seed 0 --iterations $ITER"
T="--targets full_tensor --multioutput"

# A descriptor build that died partway leaves a short descriptors.csv behind;
# train_catboost would inner-join down to it, print the reduced row count and
# hand back meaningless R^2 -- a failure that looks like a result.  Check.
for f in "$TDA" "$THR" "$BAS"; do
  n=$(( $(wc -l < "$f") - 1 ))
  [ "$n" -eq 5000 ] || { echo "ABORT: $f has $n rows, expected 5000."; exit 1; }
  echo "ok: $f  $n rows"
done

mkdir -p "$OUT"

case " $PASSES " in *" joint "*)
for g in $FEATGROUPS; do
  echo "########## full_tensor (joint) -- $g ##########"
  python scripts/train_catboost.py $D $F $CV $T $SCALE_ARG \
      --groups "$g" --output "$OUT/full_tensor__$g"
done
;; esac

# Second pass: k_off on its own.  MultiRMSE minimises the SUMMED squared error
# over the three components on their raw scales, and k_xx carries 17.8x the
# variance of k_off, so the diagonals dominate the joint loss and the
# off-diagonal comes out under-fitted -- measured at -.035 on `baselines`
# (joint .679 vs single .714, the runs this replaces).  k_off is the component
# the study turns on, so it also gets fitted alone, at full strength and
# comparable with every recorded 2D k_off number.
case " $PASSES " in *" single "*)
for g in $FEATGROUPS; do
  echo "########## k_off (single) -- $g ##########"
  python scripts/train_catboost.py $D $F $CV --targets k_off \
      --groups "$g" --output "$OUT/k_off__$g"
done
;; esac

python scripts/collect_results.py --results RESULTS --target k_off
