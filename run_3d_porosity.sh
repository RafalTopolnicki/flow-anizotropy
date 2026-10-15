#!/bin/bash
# 3D porosity-stratified fit: does the throat descriptor's margin over the
# baselines GROW as porosity falls?  Zero LBM cost -- this refits the existing
# DATA/aniso3d labels inside porosity quartiles of themselves.
#
# THE QUESTION. `thr` beats `baselines` on all six components pooled over
# porosity .650-.900. If throats are the mechanism, the margin should widen
# toward low porosity, where throats are more often the limiting constriction.
#
# THE 2D ANSWER WAS NO, and that is the reason to run this. In 2D the margin
# SHRANK toward low porosity (thr +.0341 at q4 -> -.0045 at q1) and the `tda`
# control shrank with it, so the effect was common to every descriptor family.
# It may not transfer: planar duality forbids pore and solid both percolating
# in 2D, so connectivity cannot vary much there, and connectivity is the
# proposed mechanism for `thr` winning in 3D at all. A 3D gradient of the
# opposite sign would be the direct evidence for a lower-porosity family;
# the same sign says the mechanism is dimension-independent and new structures
# would not rescue it.
#
# READ `tda` AS THE CONTROL. Inside a narrow porosity band, porosity explains
# less of k's variance, so every descriptor tends to look better. If BOTH
# margins grow it is that artefact; only `thr` growing while `tda` stays flat
# implicates throats.
#
# R^2 IS NOT COMPARABLE ACROSS BANDS. k_xy sd runs .39 (q1) -> 1.03 (q4), so
# the whole scale of k moves. Compare GROUP DIFFERENCES within a band, and
# track (group - baselines) across bands. The label-noise ceiling is
# .9999999993 on the off-diagonals, so it binds nothing in any band.
#
# Quartile cuts are on all 5000 rows (every one converged). Half-open, so each
# structure is used exactly once.
#
# COST: n=1250/band, nine groups, k_xy single -> ~1.5-2 h.
#
#   bash run_3d_porosity.sh 2>&1 | tee RUN3D_POR.log
# ---------------------------------------------------------------------------
set -u

ITER=${ITER:-1500}
FOLDS=${FOLDS:-5}
SEED=${SEED:-0}
OUT=${OUT:-RESULTS/aniso3d_por}
DATASET=${DATASET:-DATA/aniso3d}

# NOT named GROUPS -- bash reserves that name and discards assignments to it.
# The baseline families are fitted separately as well as concatenated: a
# combined row hides that `tpc` carries nearly all of it (.7497 of the
# 1819-feature union's .7725 on k_xy) while `por` is ~0 on the off-diagonals.
FEATGROUPS=${FEATGROUPS:-"por tpc fab fabdir baselines tda thr baselines+thr all"}

# Off-diagonals only by default: the diagonals are near-saturated (.95-.99 for
# every method) and separate the groups far less. TARGETS="k_xy k_xz k_yz" for
# all three, or add K_xx K_yy K_zz to sweep the whole tensor.
TARGETS=${TARGETS:-"k_xy"}

# Quartiles of porosity over the 5000 converged rows.
BANDS=${BANDS:-"q1:0.0:0.713381 q2:0.713381:0.774594 q3:0.774594:0.838269 q4:0.838269:1.0"}

FIX=DESC/aniso3d_ph_fixed/descriptors.csv     # re-imaged cone: ecp_ + ph_
THR=DESC/aniso3d_throat/descriptors.csv       # throat PH: thr_
BAS=DESC/aniso3d/baselines.csv                # por_ fabdir_ tpc_ fab_
EXPECT=${EXPECT:-5000}

for f in "$FIX" "$THR" "$BAS"; do
  if [ ! -f "$f" ]; then echo "ABORT: $f does not exist."; exit 1; fi
  n=$(( $(wc -l < "$f") - 1 ))
  if [ "$n" -ne "$EXPECT" ]; then
    echo "ABORT: $f has $n rows, expected $EXPECT."; exit 1
  fi
  echo "ok: $f  $n rows"
done

D="--dataset $DATASET --labels $DATASET/permeability.csv"
F="--descriptors $FIX $THR $BAS"
CV="--folds $FOLDS --seed $SEED --iterations $ITER"

mkdir -p "$OUT"
echo "groups   $FEATGROUPS"
echo "targets  $TARGETS   bands $BANDS"

for b in $BANDS; do
  name=${b%%:*}; rest=${b#*:}; lo=${rest%%:*}; hi=${rest##*:}
  for t in $TARGETS; do
    for g in $FEATGROUPS; do
      dir="$OUT/$name/${t}__$g"
      if [ -f "$dir/metrics.json" ]; then
        echo "########## $name $t -- $g -- already done, skipping ##########"
        continue
      fi
      echo "########## $name  porosity [$lo, $hi)  $t  --  $g  $(date -Is) ##########"
      python scripts/train_catboost.py $D $F $CV \
          --targets "$t" --porosity-range "$lo" "$hi" \
          --groups "$g" --output "$dir"
    done
  done
done

python scripts/collect_results.py --results RESULTS --target k_xy
echo "RUN3D POR DONE  $(date -Is)"
