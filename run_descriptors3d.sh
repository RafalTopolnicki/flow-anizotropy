#!/bin/bash
# Directional TDA descriptors + baselines for the 3D dataset.
#
# Per direction, in chunks of CHUNK structures:
#   filtration -> ECP (Julia) + PH (gudhi) -> drop the filtration.
#
# The chunking is not cosmetic. A filtration is 6.1 MB (80^3 x 3 channels x
# float32), so one whole direction is ~31 GB -- and /home carries a **150 GiB
# ceph quota that `df` does not report**: df shows the pool's 18 TB of free
# space, the quota is what actually applies. The first attempt at this run died
# partway through direction 1 for exactly that reason. At CHUNK=1000 peak disk
# is ~6 GB instead of 31 GB.
#
#   getfattr -n ceph.quota.max_bytes ~   # the limit
#   getfattr -n ceph.dir.rbytes      ~   # current usage
#
# Needs no labels -- run it while the LBM is still going on the other machine.
# Every stage skips samples whose output already exists, so an interrupted run
# is resumed by rerunning this script.
#
# Measured on 24 cores, 80^3, 5000 structures: ~4 h for the nine directions,
# plus ~15 min of per-chunk Julia/Python startup. See PLAN_descriptors3d.md.
# pipefail matters here: the PH stage is `python ... | tee`, whose status is
# tee's, so `set -e` alone treats a crashed ph2d/ph3d as success and the run
# continues with a silently incomplete direction.
set -eo pipefail

DATASET=${DATASET:-DATA/aniso3d}
DESCROOT=${DESCROOT:-DESC/aniso3d}
WORKERS=${WORKERS:-$(nproc)}
GRID_RES=${GRID_RES:-6}
PH_RES=${PH_RES:-"10 10"}
IM_RANGE=${IM_RANGE:-"0.0 1.25 0.0 1.25"}
RADIUS=${RADIUS:-4}
CONE_R=${CONE_R:-3}
CONE_H=${CONE_H:-6}
N_POR=${N_POR:-80}
N_TPC=${N_TPC:-41}
JLENV=${JLENV:-$HOME/direction-aware-tda-for-porous-materials}
KEEP_FILT=${KEEP_FILT:-0}
LIMIT=${LIMIT:-0}
CHUNK=${CHUNK:-1000}

# Total structures to process, and the chunk boundaries within a direction.
N_TOTAL=$(( $(wc -l < "$DATASET/structures.csv") - 1 ))
if [ "$LIMIT" != "0" ]; then N_TOTAL=$LIMIT; fi

# 3 axes + 6 signed face diagonals.  The signed pairs are the point: k_xy is the
# contrast between [1,1,0] and [1,-1,0], so a set with only one of the pair
# cannot express the sign of an off-diagonal.  The elasticity repo's 7-direction
# set has [1,1,0] and not [1,-1,0]; the 2D arm used 0/45/90/135 for this reason.
# Body diagonals ([1,1,1] etc.) are a deliberate omission -- see PLAN §3.
if [ -n "$DIRECTIONS_OVERRIDE" ]; then
  DIRECTIONS=("$DIRECTIONS_OVERRIDE")
else
  DIRECTIONS=("1 0 0" "0 1 0" "0 0 1" \
              "1 1 0" "1 -1 0" \
              "1 0 1" "1 0 -1" \
              "0 1 1" "0 1 -1")
fi

for D in "${DIRECTIONS[@]}"; do
  TAG=${D// /_}
  FILT=$DESCROOT/filt/a$TAG

  # A direction whose PH is already complete is skipped whole, so an
  # interrupted run resumes without recomputing finished directions.
  # Counted with a -d guard, not `ls DIR 2>/dev/null | wc -l`: under
  # `set -eo pipefail` that idiom is fatal, because ls on a directory that does
  # not exist yet exits non-zero, pipefail propagates it past the successful
  # `wc`, and set -e kills the script with NO message -- ls's stderr having
  # gone to /dev/null. That is exactly how stage 2 died silently on
  # 2026-09-18: the cone root had no output directories on a first run.
  if [ -d "$DESCROOT/ph/a$TAG" ]; then NDONE=$(ls -1 "$DESCROOT/ph/a$TAG" | wc -l); else NDONE=0; fi
  if [ "$NDONE" -ge "$N_TOTAL" ]; then
    echo "=== direction [$D] -- already complete ($NDONE), skipping ==="
    continue
  fi

  echo "=== direction [$D] ==="
  for START in $(seq 0 "$CHUNK" $((N_TOTAL - 1))); do
    echo "--- [$D] structures $START..$((START + CHUNK)) ---"

    python scripts/filtration3d.py --dataset "$DATASET" --outputdir "$FILT" \
        --direction $D --radius "$RADIUS" \
        --cone-radius "$CONE_R" --cone-height "$CONE_H" \
        --workers "$WORKERS" --start "$START" --limit "$CHUNK"

    julia --project="$JLENV" --threads "$WORKERS" scripts/ecp3d.jl \
        "$FILT" "$DESCROOT/ecp/a$TAG" --grid-res "$GRID_RES"

    python scripts/ph3d.py --inputdir "$FILT" --outputdir "$DESCROOT/ph/a$TAG" \
        --resolution $PH_RES --im_range $IM_RANGE --workers "$WORKERS"

    # KEEP_FILT=1 keeps every chunk and WILL exceed the quota -- debugging only,
    # and only with CHUNK small enough that one chunk fits.
    if [ "$KEEP_FILT" != "1" ]; then rm -rf "$FILT"; fi
  done
done

python scripts/collect_descriptors.py --dataset "$DATASET" \
    --descroot "$DESCROOT" --output "$DESCROOT/descriptors.csv"

# Baselines: porosity profile, two-point correlation, fabric profile (all nine
# directions) plus the global fabric tensor.  Minutes, not hours.  The fabric
# tensor is emitted as 6 components + 3 eigenvalues + fractional anisotropy --
# never eigenvector directions, which are sign-ambiguous, reorder when
# eigenvalues cross, and are degenerate for a rod or a plate (NOTES 8.5).
python scripts/baselines3d.py --dataset "$DATASET" \
    --output "$DESCROOT/baselines.csv" \
    --n_por "$N_POR" --n_tpc "$N_TPC" --workers "$WORKERS" --limit "$LIMIT"

# Labels, whenever they have landed -- joins on sample_id, so a partial perm/
# directory (723 of 5000 as of 2026-09-10, with gaps) merges cleanly.
# python scripts/solve_permeability3d.py --dataset "$DATASET" --merge-only
