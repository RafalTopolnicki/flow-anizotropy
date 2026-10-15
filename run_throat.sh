#!/bin/bash
# Step 0 of the "can TDA beat baselines" plan: throat-scale PH on the EXISTING
# labelled 5000, no new LBM.
#
# Builds two descriptor roots from scripts/filtration_throat2d.py:
#
#   DESC/aniso_throat/       --mode dir, four directions   -> thr_a{0,45,90,135}_*
#   DESC/aniso_throat_iso/   --mode iso, one tag (a0)      -> thr_a0_*
#
# The column prefix is `thr_`, deliberately NOT `ph_`: the names `ph`, `tda`,
# `baselines` and `all` index runs already recorded in RESULTS/summary.csv, and
# reusing them for a different filtration would make old and new numbers
# incomparable under the same label (the warning in train_catboost.py's
# PREFIXES, and the failure mode of NOTES 11.11).
#
# Imaging parameters below are measured from THIS filtration's own bars, per
# the rule in NOTES 11.5c / 11.11 -- they are not the wedge's values and must
# not be copied from run_descriptors.sh.  Sampled over 40 structures:
#
#   mode      H0 pers p99   H1 pers p99   H1 bars/struct
#   iso          0.354         0.815            38
#   dir a0       0.403         0.666            38
#   dir a45      0.348         0.379           944
#
# so H0 persistence is ranged to 0.45 and H1 to 0.95, and each bandwidth is the
# geometric mean of its two pixel sizes (~1 px, the target in ph2d.py).
# pipefail matters here: the PH stage is `python ... | tee`, whose status is
# tee's, so `set -e` alone treats a crashed ph2d/ph3d as success and the run
# continues with a silently incomplete direction.
set -eo pipefail

DATASET=${DATASET:-DATA/aniso}
WORKERS=${WORKERS:-$(nproc)}
RCAP=${RCAP:-28}
HORIZON=${HORIZON:-6}
PHASE=${PHASE:-void}
PH_WEIGHT=${PH_WEIGHT:-linear}
PH_RES=${PH_RES:-"20 20"}
PH_RANGE_H0=${PH_RANGE_H0:-"0 1.0 0 0.45"}
PH_RANGE_H1=${PH_RANGE_H1:-"0 1.0 0 0.95"}
PH_BW_H0=${PH_BW_H0:-0.0335}
PH_BW_H1=${PH_BW_H1:-0.0487}
KEEP_FILT=${KEEP_FILT:-0}

# one (descroot, mode, direction) triple per line
run_one () {
  local DESCROOT=$1 MODE=$2 TH=$3
  local FILT=$DESCROOT/filt/a$TH
  echo "=== $DESCROOT  mode=$MODE  theta=$TH ==="

  python scripts/filtration_throat2d.py --dataset "$DATASET" \
      --outputdir "$FILT" --mode "$MODE" --direction "$TH" \
      --rcap "$RCAP" --horizon "$HORIZON" --phase "$PHASE" \
      --workers "$WORKERS"

  mkdir -p "$DESCROOT"
  python scripts/ph2d.py --inputdir "$FILT" --outputdir "$DESCROOT/thr/a$TH" \
      --weight "$PH_WEIGHT" --resolution $PH_RES \
      --bandwidth_h0 "$PH_BW_H0" --bandwidth_h1 "$PH_BW_H1" \
      --im_range_h0 $PH_RANGE_H0 --im_range_h1 $PH_RANGE_H1 \
      --workers "$WORKERS" | tee "$DESCROOT/ph_params_a$TH.log"

  if grep -q SMEARED "$DESCROOT/ph_params_a$TH.log"; then
      echo "ABORT: ph2d reports a SMEARED persistence image for $DESCROOT a$TH."
      echo "Re-measure this filtration's bar statistics and reset PH_BW_*/PH_RANGE_*."
      exit 1
  fi

  [ "$KEEP_FILT" = "1" ] || rm -rf "$FILT"
}

for TH in 0 45 90 135; do
  run_one DESC/aniso_throat dir "$TH"
done
run_one DESC/aniso_throat_iso iso 0

for DESCROOT in DESC/aniso_throat DESC/aniso_throat_iso; do
  python scripts/collect_descriptors.py --dataset "$DATASET" \
      --descroot "$DESCROOT" --output "$DESCROOT/descriptors.csv" --kinds thr
  # The screen and the trainer both expect the descroot to carry its own
  # baselines -- hardcoding DESC/aniso/baselines.csv is the bug that cost a
  # retracted result (NOTES 11.11).  Same structures here, so a symlink is
  # honest, and it is explicit rather than implicit.
  ln -sfn ../aniso/baselines.csv "$DESCROOT/baselines.csv"
  cat > "$DESCROOT/config.txt" <<EOF
filtration scripts/filtration_throat2d.py  rcap $RCAP  horizon $HORIZON  phase $PHASE
PH weight $PH_WEIGHT | resolution $PH_RES
im_range H0 $PH_RANGE_H0   bandwidth $PH_BW_H0
im_range H1 $PH_RANGE_H1   bandwidth $PH_BW_H1
baselines.csv is a symlink to DESC/aniso/baselines.csv (same structures).
Built $(date -I) by run_throat.sh. See NOTES step 0.
EOF
done

echo "THROAT DESC DONE"
