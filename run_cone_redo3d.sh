#!/bin/bash
# Re-image the 3D CONE PH with a non-degenerate representation.
#
# Why: until 2026-09-17 `ph3d.py` called PersistenceImage with both of gudhi's
# defaults -- constant weight and bandwidth 1.0 -- so every recorded 3D `ph`
# and `tda` number rests on rank-1 persistence images. Measured on
# DESC/aniso3d, direction [1,0,0], the 100-column blocks have top-1 variance
# .9960 (H0), .9991 (H1), .9997 (H2). This is the same pair of defects found
# and fixed in 2D in session 7 (NOTES 11.5, 11.5a), never ported to 3D; in 2D
# repairing it gained +.097 on the `ph` group.
#
# There is a SECOND defect here, specific to H2 and worse than the bandwidth.
# Measured over 6 structures, cone H2 persistence is p50 0.006, p99 0.023,
# max 0.043 -- and it was imaged over p[0,1.25]. At resolution 10 a pixel is
# 0.125, so EVERY H2 bar landed in the first pixel row. H2 is the dimension
# NOTES 10.3 predicted would make PH worth computing in 3D, and it was never
# actually resolved.
#
# Measured cone bar statistics (6 structures, solid phase, the recorded
# convention -- this rebuild changes the imaging and NOTHING else):
#
#   d        H0 pers p99   H1 pers p99   H2 pers p99   H2 birth range
#   [1,0,0]     1.028         0.719         0.0215      0.513-0.981
#   [1,1,0]     0.988         0.627         0.0233      0.483-0.991
#
# So H0 keeps p[0,1.25], H1 narrows to p[0,0.80], and H2 gets its own window
# b[0.2,1.0] p[0,0.06]. Each bandwidth is the geometric mean of its two pixel
# sizes. H2's grid is unavoidably anisotropic (0.24 px birth, 3.2 px
# persistence): gudhi's PersistenceImage takes ONE bandwidth for both axes, so
# on a 13:1 window no single value is ~1 px in both. The geometric mean is the
# compromise, and the SMEARED gate in ph3d.py is what keeps it honest.
#
# Scope and cost: the same contiguous labelled prefix as run_throat3d.sh
# (--limit 2816), nine directions. ECP does not depend on PH imaging, so it is
# symlinked rather than recomputed. `--channels 1` writes only the cone channel
# that PH reads -- 2.05 MB per filtration instead of 6.1 MB, which matters
# against the ceph quota (~25 GB free).
# See run_throat3d.sh: pipefail is required because `python ... | tee` hides a
# crashed PH stage from `set -e`, and intermediates must stay off the ceph
# quota.
set -eo pipefail
TMPROOT=${TMPROOT:-/tmp/claude-607000063/-home-rtopolnicki-flow-flow-anizotropy/d2946d9b-c2e6-462e-8071-db1207ecd447/scratchpad/filt3d}
# See run_throat3d.sh: the binding quota is inodes (max_files = 1e6),
# so the per-structure .npy tree stays on /tmp and only the CSV lands
# under DESC. ecp/ is symlinked into the scratch root by absolute path
# so collect_descriptors can read both kinds from one descroot.
NPYROOT=${NPYROOT:-/tmp/claude-607000063/-home-rtopolnicki-flow-flow-anizotropy/d2946d9b-c2e6-462e-8071-db1207ecd447/scratchpad/npy}

DATASET=${DATASET:-DATA/aniso3d}
DESCROOT=${DESCROOT:-DESC/aniso3d_ph_fixed}
WORKERS=${WORKERS:-$(nproc)}
# 5000 since the LBM solve completed 2026-09-22 (was 2816, the labelled prefix).
LIMIT=${LIMIT:-5000}
# FROM: extend rather than rebuild. See run_throat3d.sh for why the .npy
# resume check cannot see an already-collected head. ECP is unaffected --
# DESC/aniso3d/ecp holds all 5000 and is symlinked, so only PH is recomputed.
FROM=${FROM:-0}
CHUNK=${CHUNK:-500}
RADIUS=${RADIUS:-4}
CONE_R=${CONE_R:-3}
CONE_H=${CONE_H:-6}
PH_WEIGHT=${PH_WEIGHT:-linear}
PH_RES=${PH_RES:-"10 10"}
KEEP_FILT=${KEEP_FILT:-0}

RANGE_H0=${RANGE_H0:-"0 1.25 0 1.25"};  BW_H0=${BW_H0:-0.125}
RANGE_H1=${RANGE_H1:-"0 1.25 0 0.80"};  BW_H1=${BW_H1:-0.100}
RANGE_H2=${RANGE_H2:-"0.2 1.0 0 0.06"}; BW_H2=${BW_H2:-0.019}

# The same 9 directions as run_descriptors3d.sh, so the rebuilt `ph` is
# column-for-column comparable with the recorded one.
DIRECTIONS=("1 0 0" "0 1 0" "0 0 1" \
            "1 1 0" "1 -1 0" \
            "1 0 1" "1 0 -1" \
            "0 1 1" "0 1 -1")

for D in "${DIRECTIONS[@]}"; do
  TAG=${D// /_}
  FILT=$TMPROOT/cone_redo/a$TAG
  PHOUT=$NPYROOT/cone_redo/ph/a$TAG

  # Counted with a -d guard, not `ls DIR 2>/dev/null | wc -l`: under
  # `set -eo pipefail` that idiom is fatal, because ls on a directory that does
  # not exist yet exits non-zero, pipefail propagates it past the successful
  # `wc`, and set -e kills the script with NO message -- ls's stderr having
  # gone to /dev/null. That is exactly how stage 2 died silently on
  # 2026-09-18: the cone root had no output directories on a first run.
  if [ -d "$PHOUT" ]; then NDONE=$(ls -1 "$PHOUT" | wc -l); else NDONE=0; fi
  # Against LIMIT-FROM, not LIMIT: with FROM>0 this directory only holds the tail.
  if [ "$NDONE" -ge $((LIMIT - FROM)) ]; then
    echo "=== cone redo [$D] -- already complete ($NDONE), skipping ==="
    continue
  fi
  echo "=== cone redo  d=[$D] ==="

  for START in $(seq "$FROM" "$CHUNK" $((LIMIT - 1))); do
    TAKE=$CHUNK
    if [ $((START + TAKE)) -gt "$LIMIT" ]; then TAKE=$((LIMIT - START)); fi
    echo "--- [$D] structures $START..$((START + TAKE)) ---"

    python scripts/filtration3d.py --dataset "$DATASET" --outputdir "$FILT" \
        --direction $D --radius "$RADIUS" \
        --cone-radius "$CONE_R" --cone-height "$CONE_H" --channels 1 \
        --workers "$WORKERS" --start "$START" --limit "$TAKE"

    mkdir -p "$DESCROOT" "$PHOUT"
    python scripts/ph3d.py --inputdir "$FILT" --outputdir "$PHOUT" \
        --weight "$PH_WEIGHT" --resolution $PH_RES \
        --im_range_h0 $RANGE_H0 --bandwidth_h0 "$BW_H0" \
        --im_range_h1 $RANGE_H1 --bandwidth_h1 "$BW_H1" \
        --im_range_h2 $RANGE_H2 --bandwidth_h2 "$BW_H2" \
        --workers "$WORKERS" | tee "$DESCROOT/ph_params_a$TAG.log"

    if grep -q SMEARED "$DESCROOT/ph_params_a$TAG.log"; then
        echo "ABORT: ph3d reports a SMEARED persistence image for a$TAG."
        exit 1
    fi

    [ "$KEEP_FILT" = "1" ] || rm -rf "$FILT"
  done
done

# ECP is untouched by PH imaging, so it is the recorded one, symlinked. That
# makes the `tda` and `all` groups in this root differ from the recorded ones
# in the PH block ONLY, which is the single variable this rebuild isolates.
mkdir -p "$DESCROOT" "$NPYROOT/cone_redo"
ln -sfn "$PWD/DESC/aniso3d/ecp" "$NPYROOT/cone_redo/ecp"
ln -sfn ../aniso3d/baselines.csv "$DESCROOT/baselines.csv"

if [ "$FROM" -eq 0 ]; then
  OUTCSV="$DESCROOT/descriptors.csv"
else
  # See run_throat3d.sh: preserve the head as its own part, never overwrite
  # descriptors.csv with a tail-only table.
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
    --descroot "$NPYROOT/cone_redo" --output "$OUTCSV" \
    --kinds ecp ph --start "$FROM" --limit "$LIMIT"

if [ "$FROM" -ne 0 ]; then
  python scripts/merge_descriptors.py \
      --parts "$DESCROOT/descriptors_part_0_$FROM.csv" "$OUTCSV" \
      --output "$DESCROOT/descriptors.csv" \
      --dataset "$DATASET" --expect "$LIMIT"
fi

cat > "$DESCROOT/config.txt" <<EOF
cone filtration (scripts/filtration3d.py), solid phase, radius $RADIUS cone $CONE_R/$CONE_H
structures: first $LIMIT of $DATASET$([ "$FROM" -ne 0 ] && echo " (rows 0..$((FROM-1)) built earlier; $FROM..$((LIMIT-1)) appended $(date -I) and joined by merge_descriptors.py -- see descriptors.csv.parts.json)")
PH weight $PH_WEIGHT | resolution $PH_RES | dims H0,H1,H2
im_range H0 $RANGE_H0   bandwidth $BW_H0
im_range H1 $RANGE_H1   bandwidth $BW_H1
im_range H2 $RANGE_H2   bandwidth $BW_H2
ecp/ and baselines.csv are symlinks to DESC/aniso3d (unchanged by this rebuild).
Built $(date -I) by run_cone_redo3d.sh. See NOTES 11.13.
EOF

echo "CONE REDO DONE"
