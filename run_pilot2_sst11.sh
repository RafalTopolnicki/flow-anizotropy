#!/bin/bash
# Second pilot: g refinement at s=5 between pilot 1's g=4 (pores too narrow) and
# g=8 (porosity ~.75). Target: dt_p50 >= ~2 voxels at porosity ~.6, as rock_berea.
# 3 rocks in parallel (the SDF of a 1000^3 rock is single-threaded and dominates).
# 1 x 3 x 5 shapes x 4 rotations = 60 cubes per rock.
#   nohup bash run_pilot2_sst11.sh > PILOT2_SST11.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")"
for rock in Parker CastleGate Berea; do
  python -u scripts/stretch_rock.py --pilot --rock $rock --scales 5 --grows 5 6 7 \
      --reps 4 --jobs ${JOBS:-8} --out DATA/rock_pilot_sst11_g/$rock > PILOT2_$rock.log 2>&1 &
done
wait
echo "PILOT2 DONE $(date -Is)"
