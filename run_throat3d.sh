#!/bin/bash
# Step 0, ported to 3D: throat-scale PH on the EXISTING 2816 labels, no new LBM.
#
# Builds two descriptor roots from scripts/filtration_throat3d.py:
#
#   DESC/aniso3d_throat/       --mode dir, nine directions -> thr_a<TAG>_*
#   DESC/aniso3d_throat_iso/   --mode iso, one tag (a1_0_0) -> thr_a1_0_0_*
#
# Column prefix `thr_`, deliberately NOT `ph_`: the names `ph`, `tda` and `all`
# index the recorded 3D runs, which are built on the CONE filtration, and
# reusing them would make old and new numbers incomparable under one label.
#
# Scope: --limit 2816.  DATA/aniso3d holds 5000 structures but only the first
# 2816 are labelled, and they are CONTIGUOUS (sample_id 0..2815, no gaps), so a
# prefix is exactly the labelled set.  That halves a 9.5 h run to ~2.6 h on 52
# cores.  collect_descriptors.py is given the same --limit or its
# missing-descriptor check trips on the unlabelled 2184.
#
# Imaging: measured from THIS filtration's own bars over 8 structures, per the
# rule in NOTES 11.5c / 11.11.  These are NOT the cone's values and must not be
# copied from run_descriptors3d.sh, which passes gudhi's defaults and thereby
# produces rank-1 images (top-1 variance .996/.9991/.9997 -- NOTES 11.13).
#
#   mode        H0 p99   H1 p99   H2 p99   H1 bars/struct
#   dir [1,0,0]  0.318    0.673    0.416        264
#   dir [1,1,0]  0.267    0.124    0.041       8651   <- diagonal, cubical artefact
#   iso          0.219    0.866    0.662        708
#
# The diagonal bar explosion is intrinsic to diagonal directions in cubical
# homology, not a rounding bug: normalising the walk to a digital line
# (max|d_i| = 1) instead of unit Euclidean steps makes it slightly WORSE
# (7593 vs 6587 H1 bars), and PH cost is set by the 80^3 complex either way.
#
# Cost, measured: filtration 0.81 s/structure, PH ~17.5 s/structure.  On 52
# cores, ~2.6 h for all ten runs.  Disk: one filtration is 2.05 MB, so a
# direction is 5.8 GB -- chunked and deleted, because /home carries a 150 GiB
# ceph quota `df` does not report and only ~25 GB of it is free.
#
#   getfattr -n ceph.quota.max_bytes ~
#   getfattr -n ceph.dir.rbytes      ~
#
# Every stage skips samples whose output exists, so an interrupted run resumes
# by rerunning this script.
# `set -o pipefail` is load-bearing, not hygiene: every PH stage below is
# `python ... | tee log`, and a bash pipeline's status is the LAST command's.
# tee always succeeds, so `set -e` alone let a crashed ph3d.py pass as success
# -- which is exactly what happened on the first attempt at this run. The PH
# stage hit the ceph quota partway through direction [1,0,-1], the driver
# marched on through two more directions, and only collect_descriptors would
# have caught it, hours later.
set -eo pipefail

# Intermediates go to a scratch filesystem OUTSIDE the ceph quota. /home is
# quota'd at 150 GiB with ~19 GiB free, and .conda alone holds 58 GiB; worse,
# ceph charges deleted files until the MDS reclaims them, so churning thirty
# 2 GB chunk directories through /home made the accounted usage climb
# monotonically (23.0 -> 21.4 -> 19.4 GiB free) even though the steady-state
# footprint is flat. /tmp is a separate 99 GB disk. Only the tiny PH outputs
# (2400 B per structure) land under DESC.
TMPROOT=${TMPROOT:-/tmp/claude-607000063/-home-rtopolnicki-flow-flow-anizotropy/d2946d9b-c2e6-462e-8071-db1207ecd447/scratchpad/filt3d}

# The BINDING quota is not bytes -- it is inodes. /home carries
# ceph.quota.max_files = 1,000,000 and sits at ~1,000,007 entries, with .conda
# alone holding 486,958 files; the byte quota had 23 GiB spare when writes
# started failing with Errno 122. This pipeline writes one .npy per structure
# per direction, so ten directions x 2816 is 28,160 inodes it simply does not
# have. The per-structure files are INTERMEDIATES -- collect_descriptors folds
# them into one CSV -- so the whole npy tree lives on /tmp and only
# descriptors.csv, config.txt and the baselines symlink land under DESC.
# Consequence to know: clearing /tmp loses the .npy cache, so a rerun
# recomputes from scratch. descriptors.csv is the durable artefact.
NPYROOT=${NPYROOT:-/tmp/claude-607000063/-home-rtopolnicki-flow-flow-anizotropy/d2946d9b-c2e6-462e-8071-db1207ecd447/scratchpad/npy}

DATASET=${DATASET:-DATA/aniso3d}
WORKERS=${WORKERS:-$(nproc)}
# 5000 since the LBM solve completed 2026-09-22 (was 2816, the labelled prefix).
LIMIT=${LIMIT:-5000}
# FROM: extend an existing table instead of rebuilding it. The per-structure
# .npy are an INTERMEDIATE -- deleted once collect_descriptors.py folds them
# into the CSV, because they are what eats the inode quota -- so the resume
# check below (which counts .npy) cannot see that structures 0..FROM-1 are
# already done and would recompute them. FROM=2816 computes only the tail and
# joins it to the head with merge_descriptors.py, which checks the seam.
FROM=${FROM:-0}
CHUNK=${CHUNK:-1000}
RCAP=${RCAP:-10}
HORIZON=${HORIZON:-6}
PHASE=${PHASE:-void}
PH_WEIGHT=${PH_WEIGHT:-linear}
PH_RES=${PH_RES:-"10 10"}
KEEP_FILT=${KEEP_FILT:-0}

# Birth extent is [0, 0.9] for every dimension: with RCAP=10 the smallest
# non-zero dt is 1, so f tops out at 1 - 1/10 = 0.9.  Persistence extents cover
# the worst p99 across directions; each bandwidth is the geometric mean of its
# two pixel sizes, which lands near 1 px as ph3d.py asks.
DIR_RANGE_H0=${DIR_RANGE_H0:-"0 0.9 0 0.40"};  DIR_BW_H0=${DIR_BW_H0:-0.060}
DIR_RANGE_H1=${DIR_RANGE_H1:-"0 0.9 0 0.75"};  DIR_BW_H1=${DIR_BW_H1:-0.082}
DIR_RANGE_H2=${DIR_RANGE_H2:-"0 0.9 0 0.50"};  DIR_BW_H2=${DIR_BW_H2:-0.067}
ISO_RANGE_H0=${ISO_RANGE_H0:-"0 0.9 0 0.30"};  ISO_BW_H0=${ISO_BW_H0:-0.052}
ISO_RANGE_H1=${ISO_RANGE_H1:-"0 0.9 0 0.95"};  ISO_BW_H1=${ISO_BW_H1:-0.092}
ISO_RANGE_H2=${ISO_RANGE_H2:-"0 0.9 0 0.75"};  ISO_BW_H2=${ISO_BW_H2:-0.082}

# 3 axes + 6 signed face diagonals, the same set as run_descriptors3d.sh. The
# signed pairs are the point: k_xy is the contrast between [1,1,0] and
# [1,-1,0], so a set with only one of the pair cannot express an off-diagonal's
# sign.
DIRECTIONS=("1 0 0" "0 1 0" "0 0 1" \
            "1 1 0" "1 -1 0" \
            "1 0 1" "1 0 -1" \
            "0 1 1" "0 1 -1")

run_one () {
  local DESCROOT=$1 MODE=$2 D=$3
  local TAG=${D// /_}
  local ROOTNAME; ROOTNAME=$(basename "$DESCROOT")
  local FILT=$TMPROOT/$ROOTNAME/a$TAG
  local PHOUT=$NPYROOT/$ROOTNAME/thr/a$TAG

  local NDONE
  # Counted with a -d guard, not `ls DIR 2>/dev/null | wc -l`: under
  # `set -eo pipefail` that idiom is fatal, because ls on a directory that does
  # not exist yet exits non-zero, pipefail propagates it past the successful
  # `wc`, and set -e kills the script with NO message -- ls's stderr having
  # gone to /dev/null. That is exactly how stage 2 died silently on
  # 2026-09-18: the cone root had no output directories on a first run.
  if [ -d "$PHOUT" ]; then NDONE=$(ls -1 "$PHOUT" | wc -l); else NDONE=0; fi
  # Against LIMIT-FROM, not LIMIT: with FROM>0 this directory only ever holds
  # the tail, so comparing with LIMIT would never call it complete and every
  # rerun would redo the whole tail.
  if [ "$NDONE" -ge $((LIMIT - FROM)) ]; then
    echo "=== $DESCROOT [$D] -- already complete ($NDONE), skipping ==="
    return 0
  fi
  echo "=== $DESCROOT  mode=$MODE  d=[$D] ==="

  local R0 R1 R2 B0 B1 B2
  if [ "$MODE" = "iso" ]; then
    R0=$ISO_RANGE_H0; R1=$ISO_RANGE_H1; R2=$ISO_RANGE_H2
    B0=$ISO_BW_H0;    B1=$ISO_BW_H1;    B2=$ISO_BW_H2
  else
    R0=$DIR_RANGE_H0; R1=$DIR_RANGE_H1; R2=$DIR_RANGE_H2
    B0=$DIR_BW_H0;    B1=$DIR_BW_H1;    B2=$DIR_BW_H2
  fi

  for START in $(seq "$FROM" "$CHUNK" $((LIMIT - 1))); do
    local TAKE=$CHUNK
    if [ $((START + TAKE)) -gt "$LIMIT" ]; then TAKE=$((LIMIT - START)); fi
    echo "--- [$D] structures $START..$((START + TAKE)) ---"

    python scripts/filtration_throat3d.py --dataset "$DATASET" \
        --outputdir "$FILT" --mode "$MODE" --direction $D \
        --rcap "$RCAP" --horizon "$HORIZON" --phase "$PHASE" \
        --workers "$WORKERS" --start "$START" --limit "$TAKE"

    mkdir -p "$DESCROOT" "$PHOUT"
    python scripts/ph3d.py --inputdir "$FILT" \
        --outputdir "$PHOUT" \
        --weight "$PH_WEIGHT" --resolution $PH_RES \
        --im_range_h0 $R0 --bandwidth_h0 "$B0" \
        --im_range_h1 $R1 --bandwidth_h1 "$B1" \
        --im_range_h2 $R2 --bandwidth_h2 "$B2" \
        --workers "$WORKERS" | tee "$DESCROOT/ph_params_a$TAG.log"

    if grep -q SMEARED "$DESCROOT/ph_params_a$TAG.log"; then
        echo "ABORT: ph3d reports a SMEARED persistence image for $DESCROOT a$TAG."
        echo "Re-measure this filtration's bars and reset the ranges/bandwidths."
        exit 1
    fi

    # KEEP_FILT=1 WILL exceed the quota at nine directions -- debugging only.
    [ "$KEEP_FILT" = "1" ] || rm -rf "$FILT"
  done
}

for D in "${DIRECTIONS[@]}"; do
  run_one DESC/aniso3d_throat dir "$D"
done
run_one DESC/aniso3d_throat_iso iso "1 0 0"

for DESCROOT in DESC/aniso3d_throat DESC/aniso3d_throat_iso; do
  if [ "$FROM" -eq 0 ]; then
    OUTCSV="$DESCROOT/descriptors.csv"
  else
    # Preserve the head as its own part BEFORE collecting, and never write the
    # tail to descriptors.csv: the head's rows are not recomputable here (their
    # .npy are gone), so an overwrite would destroy them silently.
    HEAD="$DESCROOT/descriptors_part_0_$FROM.csv"
    if [ ! -f "$HEAD" ]; then
      NHEAD=$(( $(wc -l < "$DESCROOT/descriptors.csv") - 1 ))
      if [ "$NHEAD" -ne "$FROM" ]; then
        echo "ABORT: $DESCROOT/descriptors.csv has $NHEAD rows, but FROM=$FROM"
        echo "says the head should be $FROM rows. Refusing to guess the seam."
        exit 1
      fi
      mv "$DESCROOT/descriptors.csv" "$HEAD"
    fi
    OUTCSV="$DESCROOT/descriptors_part_${FROM}_${LIMIT}.csv"
  fi

  python scripts/collect_descriptors.py --dataset "$DATASET" \
      --descroot "$NPYROOT/$(basename "$DESCROOT")" \
      --output "$OUTCSV" \
      --kinds thr --start "$FROM" --limit "$LIMIT"

  if [ "$FROM" -ne 0 ]; then
    python scripts/merge_descriptors.py \
        --parts "$DESCROOT/descriptors_part_0_$FROM.csv" "$OUTCSV" \
        --output "$DESCROOT/descriptors.csv" \
        --dataset "$DATASET" --expect "$LIMIT"
  fi
  # The descroot carries its own baselines -- hardcoding another root's is the
  # bug that cost a retracted result (NOTES 11.11). Same structures here, so a
  # symlink is honest and explicit.
  ln -sfn ../aniso3d/baselines.csv "$DESCROOT/baselines.csv"
  cat > "$DESCROOT/config.txt" <<EOF
filtration scripts/filtration_throat3d.py  rcap $RCAP  horizon $HORIZON  phase $PHASE
structures: first $LIMIT of $DATASET$([ "$FROM" -ne 0 ] && echo " (rows 0..$((FROM-1)) built earlier; $FROM..$((LIMIT-1)) appended $(date -I) and joined by merge_descriptors.py -- see descriptors.csv.parts.json)")
PH weight $PH_WEIGHT | resolution $PH_RES | dims H0,H1,H2
baselines.csv is a symlink to DESC/aniso3d/baselines.csv (same structures).
Built $(date -I) by run_throat3d.sh. See NOTES 11.13.
EOF
done

echo "THROAT3D DESC DONE"
