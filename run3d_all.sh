#!/bin/bash
# The whole 3D step-0 pipeline, serial: throat build -> cone re-image -> train.
# Serial on purpose -- each stage saturates 52 cores, so overlapping them only
# halves each. All three stages resume, so rerunning this script after an
# interruption picks up where it stopped.
#
# EXTENDING an existing build rather than redoing it:
#
#     TMPROOT=/scratch/filt3d NPYROOT=/scratch/npy FROM=2816 ./run3d_all.sh
#
# FROM is inherited by all three stages. It computes only structures
# [FROM, LIMIT) and joins them onto the descriptor tables already on disk with
# merge_descriptors.py, which refuses a seam that does not look like one
# distribution. Set it to the row count of the existing descriptors.csv. ECP is
# never recomputed either way -- DESC/aniso3d/ecp holds all 5000 already.
#
# TMPROOT/NPYROOT default to a session scratchpad that no longer exists; set
# them to real scratch or the per-structure .npy land on /home and hit the
# inode quota (1e6 files, ~784k used).
set -eo pipefail
cd /home/rtopolnicki/flow/flow-anizotropy

echo "=== STAGE 1/3 throat build  $(date -Is) ==="
./run_throat3d.sh

echo "=== STAGE 2/3 cone re-image  $(date -Is) ==="
./run_cone_redo3d.sh

echo "=== STAGE 3/3 training  $(date -Is) ==="
./train3d_throat.sh

echo "=== ALL3D STEP0 DONE  $(date -Is) ==="
