#!/bin/bash
# Third pilot: s=3 (pilot 2 showed s=5 cannot give porosity ~.6 with dt_p50 >= 2;
# smaller s widens pores in voxels at the same porosity, and s=3 keeps ~5 grains
# across the box like rock_berea at s=2). g in {4,5,6}; three new rocks to see the
# spread across rocks. 1 x 3 x 5 shapes x 4 rotations = 60 cubes per rock.
#   nohup bash run_pilot3_sst11.sh > PILOT3_SST11.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")"
for rock in Parker CastleGate Berea Bentheimer Kirby BanderaGray; do
  python -u scripts/stretch_rock.py --pilot --rock $rock --scales 3 --grows 4 5 6 \
      --reps 4 --jobs ${JOBS:-6} --out DATA/rock_pilot_sst11_s3/$rock > PILOT3_$rock.log 2>&1 &
done
wait
echo "PILOT3 DONE $(date -Is)"
