#!/bin/bash
# ---------------------------------------------------------------------------
# Porosity-stratified k_off, to decide whether a near-threshold dataset is
# worth building.
#
# THE QUESTION.  The throat descriptor filters on the Euclidean distance
# transform, so it measures throats -- and throats only become rate-limiting
# near the percolation threshold.  The 2D data lives at porosity .65-.90, where
# pores are wide and flow is not throat-controlled, and there `thr` only draws
# with the baselines (.7060 vs .7142, p=.57).  If the mechanism is real, thr's
# margin over the baselines should GROW as porosity falls.
#
# THE CONTROL, which is the point of running six groups and not two.  Inside a
# narrow porosity band, porosity itself explains less of k's variance, so the
# porosity-heavy baselines lose their easiest signal and every descriptor looks
# better.  That is an artefact of narrowing, not a throat effect.  `tda` is the
# discriminator: it is a cone/wedge descriptor with no throat semantics, so
#   - thr margin grows AND tda margin flat  -> throat mechanism, build the data
#   - both margins grow together            -> narrowing artefact, do not
# Read the two gradients together or the result means nothing.
#
# WHAT IS NOT COMPARABLE.  R^2 ACROSS bands.  The quartiles are near-equal in
# width (.060-.065) and n (1247-1248), but k_off's spread still runs 37.9 ->
# 162.1 sd from q1 to q4, so the low bands are intrinsically harder targets.
# Only compare feature groups WITHIN a band; the quantity of interest is the
# per-band difference (group - baselines) and how it moves across bands.
#
# WHY FIT INSIDE THE BAND rather than slice pooled out-of-fold predictions:
# fitting inside the band is the analogue of building a dataset at that
# porosity, which is the decision this is meant to inform.
#
# Quartile cuts are on the 4991 CONVERGED rows, and --porosity-range is applied
# after the same convergence filter, so the bands come out at 1248/1247/1248/1248.
# Open intervals below, so q1 takes everything up to the first cut and q4
# everything above the third: all 4991 rows are used exactly once.
#
# COST: ~1248 rows/band, roughly a quarter of the pooled fits -> ~15 min per
# band for six groups, ~1 h total.
#
#   bash run_2d_porosity.sh 2>&1 | tee RUN2D_POR.log
# ---------------------------------------------------------------------------
set -u

ITER=${ITER:-1500}
OUT=${OUT:-RESULTS/aniso2d_por}
# NOT named GROUPS -- bash reserves that name and discards assignments to it.
# The three baseline families are fitted SEPARATELY as well as concatenated:
# `baselines` is por_(512) + tpc_(260) + fab_(7), and a combined row alone
# hides which of the three carries the signal. Standing instruction from
# the user 2026-09-22 -- always report the components alongside the union.
FEATGROUPS=${FEATGROUPS:-"por tpc fab baselines tda thr all baselines+thr all+thr"}
BANDS=${BANDS:-"q1:0.0:0.715149 q2:0.715149:0.778137 q3:0.778137:0.839478 q4:0.839478:1.0"}

TDA=DESC/aniso_ph_bw125/descriptors.csv
THR=DESC/aniso_throat/descriptors.csv
BAS=DESC/aniso/baselines.csv

D="--dataset DATA/aniso --labels DATA/aniso/permeability.csv"
F="--descriptors $TDA $THR $BAS"
CV="--folds 5 --seed 0 --iterations $ITER"

for f in "$TDA" "$THR" "$BAS"; do
  n=$(( $(wc -l < "$f") - 1 ))
  [ "$n" -eq 5000 ] || { echo "ABORT: $f has $n rows, expected 5000."; exit 1; }
done

for b in $BANDS; do
  name=${b%%:*}; rest=${b#*:}; lo=${rest%%:*}; hi=${rest##*:}
  for g in $FEATGROUPS; do
    echo "########## $name  porosity [$lo, $hi)  --  $g ##########"
    python scripts/train_catboost.py $D $F $CV --targets k_off \
        --porosity-range "$lo" "$hi" \
        --groups "$g" --output "$OUT/$name/k_off__$g"
  done
done

python scripts/collect_results.py --results RESULTS --target k_off
