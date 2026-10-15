#!/bin/bash
# ---------------------------------------------------------------------------
# k0-stratified k_off: how the feature groups' relative standing moves with
# the number of pores per box.  Companion to run_2d_porosity.sh, same shape.
#
# WHAT k0 IS.  gen_aniso_structures.py builds the field on coordinates
# normalised to [0,1) with INTEGER wavevectors (`_build_grid`, l.93, and
# `sample_wavevectors`, l.119), so k0 is cycles per box and the geometry is
# defined in box units.  Pores per box goes like k0^2 and pore size in pixels
# like N/k0.  Over the 4991 converged rows k0 spans 4.0-10.0 -> a 6.25x range
# in pores per box, i.e. a 2.5x effective linear box size at fixed pore size.
#
# WHY THIS IS NOT (QUITE) A SYSTEM-SIZE SWEEP -- measured 2026-09-19, read this
# before quoting the result as one.  At fixed N, more pores per box necessarily
# means fewer pixels per pore; the two axes cannot be separated here.  Only a
# real `--size 600` run at fixed k0 moves pixels-per-pore alone.  Worse, the
# self-averaging mechanism that would make a box-size sweep interesting is
# ABSENT from this family: CV(k_xx) is 0.9718 / 0.9799 / 0.9785 / 0.9552 across
# q1..q4 (ratio 1.017) against the 1/sqrt(Npore) prediction of 1.93, and
# |k_off|/trace only moves .1448 -> .1317.  The cause is that `n_modes` is drawn
# at 20-31 INDEPENDENTLY of k0, so raising k0 adds pores but not independent
# degrees of freedom.  Testing self-averaging needs n_modes ~ k0^2 in the
# generator -- a new dataset, not a re-analysis.
#
# SO WHAT THIS DOES TEST: pore size in pixels, at fixed porosity and fixed
# disorder dimensionality.  Measured over 30 structures/band, the pore distance
# transform median falls 9.22 -> 7.31 -> 5.99 -> 4.73 px from q1 to q4 (p99
# 27.1 -> 14.3, px per period 53.4 -> 27.6).  `thr` filters on exactly that
# transform, so if it degrades toward q4 the cause is quantisation of the
# throat scale -- which is direct evidence that `thr` would GAIN at 600x600,
# where the transform gets 2.34x finer.  That is the decision this informs.
#
# CONFOUNDS, both small but both real:
#   - RCAP=28 px is a fixed absolute cap (filtration_throat2d.py l.63) and bites
#     unequally: 1.66% of pore pixels capped in q1, 0.018% in q4.  Small enough
#     to ignore here; it would NOT be at 600x600, where dt p99 reaches ~65 px.
#   - tpc's max_steps=128 (baselines2d.py l.69) covers 2.4 periods in q1 and
#     4.6 in q4, so `tpc` sees relatively MORE of the correlation structure in
#     the high bands.  Runs opposite to the thr quantisation effect.
# Porosity is NOT a confound: band means .7813/.7752/.7779/.7742, corr(k0,por)
# = -.026.
#
# WHAT IS NOT COMPARABLE: R^2 ACROSS bands.  k_off sd runs 169.4 -> 36.1 from
# q1 to q4 (k ~ pore area ~ k0^-2, so the whole scale of k falls).  Compare
# feature groups WITHIN a band; the quantity of interest is (group-baselines)
# per band and how that moves.  The label-noise ceiling is >= .99999 in every
# band, so it binds nothing.
#
# Quartile cuts are on the 4991 CONVERGED rows, matching run_2d_porosity.sh.
# Half-open intervals, so all 4991 rows are used exactly once.
#
# COST: ~1248 rows/band, six groups, k_off single -> ~1 h total.
#
#   bash run_2d_k0.sh 2>&1 | tee RUN2D_K0.log
# ---------------------------------------------------------------------------
set -u

ITER=${ITER:-1500}
OUT=${OUT:-RESULTS/aniso2d_k0}
# NOT named GROUPS -- bash reserves that name and discards assignments to it.
# The three baseline families are fitted SEPARATELY as well as concatenated:
# `baselines` is por_(512) + tpc_(260) + fab_(7), and a combined row alone
# hides which of the three carries the signal. Standing instruction from
# the user 2026-09-22 -- always report the components alongside the union.
FEATGROUPS=${FEATGROUPS:-"por tpc fab baselines tda thr all baselines+thr all+thr"}
BANDS=${BANDS:-"q1:0.0:5.601837 q2:5.601837:7.054182 q3:7.054182:8.565733 q4:8.565733:100.0"}

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
    echo "########## $name  k0 [$lo, $hi)  --  $g ##########"
    python scripts/train_catboost.py $D $F $CV --targets k_off \
        --filter-column k0 "$lo" "$hi" \
        --groups "$g" --output "$OUT/$name/k_off__$g"
  done
done

python scripts/collect_results.py --results RESULTS --target k_off
