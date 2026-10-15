#!/bin/bash
# s x g pilot of the stretched-cube recipe on three 11 Sandstones rocks (2.25 um native).
# Waits for scripts/fetch_sandstones11.sh. Grid: s 3..6 x g 0..12 native voxels,
# shapes (1,1) (1.5,1) (2,1) (3,1) (2,2), 2 rotations each -> 160 cubes per rock.
#   nohup bash run_pilot_sst11.sh > PILOT_SST11.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")"
until grep -q "FETCH DONE" FETCH_SST11.log; do sleep 30; done
for rock in Parker CastleGate Berea; do
  echo "########## $rock $(date -Is) ##########"
  python -u scripts/stretch_rock.py --pilot --rock $rock --scales 3 4 5 6 \
      --grows 0 4 8 12 --reps 2 --jobs ${JOBS:-24} --out DATA/rock_pilot_sst11/$rock
done
echo "PILOT DONE $(date -Is)"
