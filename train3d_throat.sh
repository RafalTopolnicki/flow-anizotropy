#!/bin/bash
# Training for the 3D step-0 port. Off-diagonals FIRST: k_xy/k_xz/k_yz are
# where TDA loses to the baselines (-.0197 mean) and where the decision lies;
# the diagonals are nearly saturated for every method (.95-.98) and separate
# them far less. Recorded 3D runs use 5 folds / seed 0 / 1500 iters / depth 6 /
# lr .05 at n=2816, so these are exactly paired with them.
set -e
cd /home/rtopolnicki/flow/flow-anizotropy

THR=DESC/aniso3d_throat/descriptors.csv
FIX=DESC/aniso3d_ph_fixed/descriptors.csv
BAS=DESC/aniso3d/baselines.csv
# 5000 since the LBM solve completed 2026-09-22. Was 2816 while the dataset
# was partially labelled; override with EXPECT=... to train on a prefix.
EXPECT=${EXPECT:-5000}

# Guard: both roots were smoke-tested at --limit 8 before the real run, so a
# build that dies partway leaves `set -e` to skip collect_descriptors and the
# 8-row descriptors.csv survives. train_catboost would then inner-join to 8
# rows, print "8 rows", and hand back meaningless R^2 -- a silent failure that
# looks like a result. Check the row count instead of trusting the file.
for F in "$THR" "$FIX"; do
  if [ ! -f "$F" ]; then
    echo "ABORT: $F does not exist -- its build did not finish."; exit 1
  fi
  N=$(( $(wc -l < "$F") - 1 ))
  if [ "$N" -ne "$EXPECT" ]; then
    echo "ABORT: $F has $N rows, expected $EXPECT."
    echo "Its build did not complete; rerun the build script (both resume)."
    exit 1
  fi
  echo "ok: $F has $N rows"
done

for T in k_xy k_xz k_yz K_xx K_yy K_zz; do
  echo "########## $T -- throat groups ##########"
  python scripts/train_catboost.py --dataset DATA/aniso3d \
      --descriptors "$THR" "$BAS" \
      --groups thr por tpc fab fabdir baselines tpc+thr baselines+thr \
      --targets $T --folds 5 --seed 0 \
      --output RESULTS/aniso3d_throat/$T

  echo "########## $T -- re-imaged cone groups ##########"
  python scripts/train_catboost.py --dataset DATA/aniso3d \
      --descriptors "$FIX" "$BAS" \
      --groups ph ecp tda all \
      --targets $T --folds 5 --seed 0 \
      --output RESULTS/aniso3d_conefixed/$T
done
echo "TRAIN3D DONE"
