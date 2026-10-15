#!/bin/bash
# LBM permeability tensors for the 11 Sandstones multi-rock set (DATA/rock_sst11,
# 4400 cubes = 11 rocks x 400, 80^3), split in TWO halves for two machines.
#
#   machine A:  nohup bash run_lbm_sst11.sh 0 > LBM_SST11_part0.log 2>&1 &
#   machine B:  nohup bash run_lbm_sst11.sh 1 > LBM_SST11_part1.log 2>&1 &
#   any time:   bash lbm_sst11_status.sh
#   when both are done:  bash merge_lbm_sst11.sh  [see that script if the disks differ]
#
# Part 0 = even sample_id, part 1 = odd: 2200 cubes each, 200 from every rock.
# This script only SOLVES; it never writes permeability.csv, so the two parts can
# run at the same time on a shared disk without stepping on each other.
#
# RESUMABLE: rerun the same command (same part number) after a kill or reboot;
# solved cubes are skipped (each is written atomically to DATA/rock_sst11/perm/<id>.csv).
#
# Steps
#   1. preflight: dataset complete, python deps
#   2. --verify --repair for this part only (another part's in-flight files are left alone)
#   3. solve this part, round-robin over rocks (--interleave rock), so a run cut
#      short still has labels for every rock -- needed for leave-one-rock-out
#
# Overrides (env):
#   WORKERS=190      parallel solves (default: all cores; one solve = one core)
#   PY=python        interpreter with numpy/pandas/tqdm (e.g. a conda env's python)
#
# Uses the existing solver LMB3d/lbm3d-perm as is (the driver's default); no build.
set -u
cd "$(dirname "$(readlink -f "$0")")"

PART=${1:-}
[ "$PART" = 0 ] || [ "$PART" = 1 ] || { echo "usage: bash run_lbm_sst11.sh 0|1"; exit 1; }

DATASET=DATA/rock_sst11
EXPECT=4400
PY=${PY:-python}
WORKERS=${WORKERS:-$(nproc)}

echo "########## LBM rock_sst11 part $PART/2 on $(hostname -s), $WORKERS workers  $(date -Is) ##########"

# ---- 1. preflight -----------------------------------------------------------
n=$(( $(wc -l < $DATASET/structures.csv) - 1 ))
[ "$n" -eq "$EXPECT" ] || { echo "ABORT: structures.csv has $n rows, expected $EXPECT"; exit 1; }
nf=$(ls $DATASET/structures | wc -l)
[ "$nf" -eq "$EXPECT" ] || { echo "ABORT: $nf structure files, expected $EXPECT"; exit 1; }
$PY -c "import numpy, pandas, tqdm" || { echo "ABORT: $PY lacks numpy/pandas/tqdm (set PY=...)"; exit 1; }

# ---- 2. audit + clear what a kill left (this part only) -------------------------
$PY scripts/solve_permeability3d.py --dataset $DATASET --shard $PART/2 --verify --repair

# ---- 3. solve this part ----------------------------------------------------------
$PY -u scripts/solve_permeability3d.py --dataset $DATASET --shard $PART/2 \
    --workers "$WORKERS" --interleave rock
rc=$?
[ $rc -eq 0 ] || echo "WARNING: solve pass exited $rc -- rerun the same command to retry failures"
echo "LBM SST11 PART $PART DONE $(date -Is)"
