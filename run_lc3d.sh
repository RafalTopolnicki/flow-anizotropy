#!/bin/bash
# 3D sample-size study: R^2 on k_xy as a function of the number of training
# structures, per descriptor group, all scored on ONE frozen hold-out.
#
#   bash run_lc3d.sh 2>&1 | tee LC3D.log
#   python scripts/lc_table.py RESULTS/lc3d_kxy/es RESULTS/lc3d_kxy/fixed
#
# HOLD-OUT. splits/aniso3d_golden.{csv,json}: 999 structures (333 per stratum,
# seed 12345), created 2026-09-23 and never trained on. learning_curve.py checks
# the ids against the sha256 in the manifest and refuses to run if they
# changed, so every size, seed, group and arm is scored on exactly the same 999
# structures -- the curves are PAIRED, and the hold-out's own sampling noise
# cancels in the differences between groups.
#
# TRAINING SUBSETS come from the other 4001. Within a seed the size-n subset is
# a prefix of the size-2n one (nested) and every prefix is balanced across the
# three strata. Several seeds per size because at n=125 one draw is mostly
# luck; the spread lc_table.py reports is across seeds.
#
# GROUPS. The five asked for -- baselines, tda, thr, baselines+tda,
# baselines+thr -- plus por/tpc/fab/fabdir separately, per the standing
# instruction of 2026-09-22 (the `baselines` union hides that `tpc` does most
# of its work). `baselines+tda` is the existing `all` group (ecp_ ph_ por_ tpc_
# fab_ fabdir_); it runs as `all` so it stays comparable with every other 3D
# table, and lc_table.py prints it as baselines+tda.
#
# TWO ARMS, because the groups differ ~800x in dimension (fab 10, all 7606):
#   es     early stopping on an inner 20% validation split carved from the
#          TRAINING subset (never the hold-out). The n=4000 point therefore
#          fits on 3200 rows and stops on 800.
#   fixed  1500 iterations flat, no validation split.
# A fixed budget overfits the wide groups hardest at small n, i.e. it biases
# against TDA exactly where data efficiency is being measured. If the ranking
# flips between the arms, that is a capacity effect, not an information one.
#
# DESCRIPTORS, same roots as run_catboost3d_all.sh: the RE-IMAGED cone
# (aniso3d_ph_fixed -- NOT aniso3d, whose ph_ carries the rank-1 bandwidth bug),
# throat PH, baselines.
#
# COST, from the single-target pass at n=4000/fit (seconds per fit: all 242,
# tda 186, baselines+thr 145, thr 87, baselines 59, por/fabdir 24, tpc 13,
# fab 2 -- ~780 s per seed at full size, ~2x that summed over the sizes):
#   fixed ~2.2 h, es somewhat less (it stops early and fits on 80%), ~4 h both.
#
# RESUMABLE: runs already in <out>/runs.csv are skipped; interrupt and rerun.
# CatBoost takes all cores; do not run two of these at once.
# ---------------------------------------------------------------------------
set -u

TARGET=${TARGET:-k_xy}
OUT=${OUT:-RESULTS/lc3d_kxy}
ARMS=${ARMS:-"es fixed"}
SIZES=${SIZES:-"125 250 500 1000 2000 4000"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
ITER=${ITER:-1500}
FEATGROUPS=${FEATGROUPS:-"baselines tda thr all baselines+thr por tpc fab fabdir"}
DATASET=${DATASET:-DATA/aniso3d}
SPLIT=${SPLIT:-splits/aniso3d_golden.csv}
EXPECT=${EXPECT:-5000}

FIX=DESC/aniso3d_ph_fixed/descriptors.csv     # re-imaged cone: ecp_ + ph_
THR=DESC/aniso3d_throat/descriptors.csv       # throat PH: thr_
BAS=DESC/aniso3d/baselines.csv                # por_ fabdir_ tpc_ fab_

for f in "$FIX" "$THR" "$BAS" "$SPLIT"; do
  if [ ! -f "$f" ]; then echo "ABORT: $f does not exist."; exit 1; fi
done
for f in "$FIX" "$THR" "$BAS"; do
  n=$(( $(wc -l < "$f") - 1 ))
  if [ "$n" -ne "$EXPECT" ]; then
    echo "ABORT: $f has $n rows, expected $EXPECT."; exit 1
  fi
  echo "ok: $f  $n rows"
done

echo "target $TARGET   groups $FEATGROUPS"
echo "sizes  $SIZES   seeds $SEEDS   arms $ARMS   iters $ITER"

for arm in $ARMS; do
  case $arm in
    es)    ES="" ;;
    fixed) ES="--no-early-stop" ;;
    *)     echo "ABORT: unknown arm '$arm' (es|fixed)"; exit 1 ;;
  esac
  echo "########## arm $arm -> $OUT/$arm  $(date -Is) ##########"
  python scripts/learning_curve.py \
      --dataset "$DATASET" --labels "$DATASET/permeability.csv" \
      --descriptors "$FIX" "$THR" "$BAS" \
      --split "$SPLIT" \
      --groups $FEATGROUPS --targets "$TARGET" \
      --sizes $SIZES --seeds $SEEDS --iterations "$ITER" $ES \
      --output "$OUT/$arm" || { echo "ABORT: arm $arm failed"; exit 1; }
done

python scripts/lc_table.py $(for arm in $ARMS; do echo "$OUT/$arm"; done) \
    --target "$TARGET" --out "$OUT/table.md"
echo "LC3D DONE  $(date -Is)"
