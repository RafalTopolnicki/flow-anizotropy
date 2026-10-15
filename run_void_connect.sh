#!/bin/bash
# Two questions, one target, settings identical to the recorded k12__* runs so
# every number here is directly comparable with RESULTS/summary.csv:
#
#   A  connectivity   do the non-additive pore features (NOTES 11.4) beat the
#                     baselines' .714 on k_off, on the real labels?
#   B  void TDA       does the same directional TDA, recomputed on the PORE
#                     phase instead of the solid skeleton, beat solid TDA's
#                     .654 / .661 -- and does `all` move off .754?
#   C  both           the flow-adapted feature set: void TDA + connectivity
#                     + baselines.
#
# Reference, solid phase, same CV (RESULTS/catboost/k12__*):
#   all .7545   baselines .7141   tpc .6842   tda .6544   ecp .6410   ph .5018
#   (ph .5990 and tda .6607 after the bandwidth fix, RESULTS/ph_bandwidth/)
set -e
cd /home/rtopolnicki/flow/flow-anizotropy

CV="--folds 5 --seed 0 --iterations 1500 --depth 6 --lr 0.05"
D="--dataset DATA/aniso --labels DATA/aniso/permeability.csv"
SOLID=DESC/aniso/descriptors.csv
VOID=DESC/aniso_void/descriptors.csv
BASE=DESC/aniso/baselines.csv
CONN=DESC/aniso/connect.csv

run () {  # run <outroot> <group> <descriptor files...>
  local out=$1 grp=$2; shift 2
  echo "=== $out / $grp ==="
  python scripts/train_catboost.py $D $CV --targets k12 --groups "$grp" \
      --descriptors "$@" --output "RESULTS/$out/k12__${grp//+/_}"
}

# ---- A: connectivity on the existing solid descriptors ---------------------
for g in conn baselines+conn all+conn tda+conn; do
  run connect "$g" $SOLID $BASE $CONN
done

# ---- B: the void-phase TDA -------------------------------------------------
for g in ecp ph tda all; do
  run void "$g" $VOID $BASE
done

# ---- C: both ---------------------------------------------------------------
for g in tda+conn all+conn; do
  run void_connect "$g" $VOID $BASE $CONN
done

echo "DONE"
