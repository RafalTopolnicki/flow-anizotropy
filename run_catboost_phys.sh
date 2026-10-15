#!/bin/bash
# CatBoost on the PHYSICS baselines (scripts/physbaselines3d.py), both 3D sets:
#
#   kc              Kozeny-Carman: phi, S, phi^3/S^2, fabric-weighted KC tensor
#   cp              directional critical radius r_c(w) + Katz-Thompson r_c^2 phi^2
#   phys            kc + cp
#   phys+thr        does throat PH add anything on top of the physics?
#   classical       baselines (por tpc fab fabdir) + kc + cp: the strongest
#                   non-topological union -- the row thr now has to beat
#   classical+thr   ... and on top of it
#   classical+tda   cone PH on top of classical (+thr: everything)
#
# Same folds, seed, iterations and single-target mode as the recorded tables
# (RESULTS/rock_berea/catboost, RESULTS/aniso3d_all), so every new row pairs
# fold-for-fold with the old ones. Output goes next to them:
#   RESULTS/rock_berea/catboost/<T>__<group>, RESULTS/aniso3d_all/<T>__<group>
# Resumable: a run whose metrics.json exists is skipped.
#
#   nohup ./run_catboost_phys.sh > CB_PHYS.log 2>&1 &
#
# Overrides: SETS="rock" FEATGROUPS="kc" ./run_catboost_phys.sh
set -u
cd "$(dirname "$(readlink -f "$0")")"

ITER=${ITER:-1500}
FOLDS=${FOLDS:-5}
SEED=${SEED:-0}
SETS=${SETS:-"rock synth"}
FEATGROUPS=${FEATGROUPS:-"kc cp phys phys+thr classical classical+thr classical+tda classical+tda+thr"}

for set in $SETS; do
  if [ "$set" = rock ]; then
    DATASET=DATA/rock_berea; OUT=RESULTS/rock_berea/catboost; EXPECT=1000
    TARGETS=${TARGETS_ROCK:-"K_xx k_xy k_xz k_yz"}
    FILES="DESC/rock_berea_cone/descriptors.csv DESC/rock_berea_throat/descriptors.csv
           DESC/rock_berea_cone/baselines.csv DESC/rock_berea_phys/physbaselines.csv"
  else
    DATASET=DATA/aniso3d; OUT=RESULTS/aniso3d_all; EXPECT=5000
    TARGETS=${TARGETS_SYNTH:-"K_xx K_yy K_zz k_xy k_xz k_yz"}
    FILES="DESC/aniso3d_ph_fixed/descriptors.csv DESC/aniso3d_throat/descriptors.csv
           DESC/aniso3d/baselines.csv DESC/aniso3d_phys/physbaselines.csv"
  fi

  # a short CSV would inner-join down to its rows and look like a result
  for f in $FILES; do
    if [ ! -f "$f" ]; then echo "ABORT: $f does not exist."; exit 1; fi
    n=$(( $(wc -l < "$f") - 1 ))
    if [ "$n" -ne "$EXPECT" ]; then echo "ABORT: $f has $n rows, expected $EXPECT."; exit 1; fi
    echo "ok: $f  $n rows"
  done

  for t in $TARGETS; do
    for g in $FEATGROUPS; do
      dir="$OUT/${t}__$g"
      if [ -f "$dir/metrics.json" ]; then
        echo "########## $set $t -- $g -- already done, skipping ##########"; continue
      fi
      echo "########## $set $t -- $g  $(date -Is) ##########"
      python scripts/train_catboost.py --dataset $DATASET --labels $DATASET/permeability.csv \
          --descriptors $FILES --folds $FOLDS --seed $SEED --iterations $ITER \
          --targets "$t" --groups "$g" --output "$dir"
    done
  done
done
echo "CB PHYS DONE  $(date -Is)"
