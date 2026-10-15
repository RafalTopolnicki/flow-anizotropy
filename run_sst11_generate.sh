#!/bin/bash
# 11 Sandstones multi-rock dataset: pilot 4 -> per-rock g -> 400 cubes per rock -> merge.
# Does NOT run the LBM (that goes on another machine).
#
#   1. pilot 4 at s=3 on the rocks pilot 3 did not cover (g 3-6) and on Bandera Gray
#      (g 2-3, it is already at porosity .67 with g=4)
#   2. scripts/choose_grow.py: per rock, the g whose porosity is closest to .60 among
#      the g where every pilot cube percolates -> DATA/rock_sst11_grow.json
#   3. stretch_rock.py --generate per rock into DATA/rock_sst11/parts/<rock>,
#      sample_id = rock index * 400 + i, then --merge into DATA/rock_sst11
#
#   nohup bash run_sst11_generate.sh > SST11_GEN.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")"
N=${N:-400}
ROCKS="BanderaBrown BanderaGray Bentheimer Berea BSG BUG BB CastleGate Kirby Leopard Parker"

echo "########## pilot 4  $(date -Is) ##########"
for rock in BanderaBrown BSG BUG BB Leopard; do
  python -u scripts/stretch_rock.py --pilot --rock $rock --scales 3 --grows 3 4 5 6 \
      --reps 4 --jobs 6 --out DATA/rock_pilot_sst11_s3b/$rock > PILOT4_$rock.log 2>&1 &
done
python -u scripts/stretch_rock.py --pilot --rock BanderaGray --scales 3 --grows 2 3 \
    --reps 4 --jobs 6 --out DATA/rock_pilot_sst11_s3b/BanderaGray > PILOT4_BanderaGray.log 2>&1 &
wait

echo "########## choose g  $(date -Is) ##########"
python scripts/choose_grow.py DATA/rock_pilot_sst11_s3 DATA/rock_pilot_sst11_s3b \
    --out DATA/rock_sst11_grow.json || exit 1

echo "########## generate  $(date -Is) ##########"
k=0
for rock in $ROCKS; do
  g=$(python -c "import json; print(json.load(open('DATA/rock_sst11_grow.json'))['rocks']['$rock']['grow'])")
  echo "$rock: g=$g ids $((k * N))..$((k * N + N - 1))"
  python -u scripts/stretch_rock.py --generate --rock $rock --n $N --scale 3 --grow $g \
      --id-offset $((k * N)) --jobs 8 --out DATA/rock_sst11/parts/$rock > GEN_$rock.log 2>&1 &
  k=$((k + 1))
  # at most 6 rocks at once (each holds a 1000^3 SDF)
  while [ "$(jobs -r | wc -l)" -ge 6 ]; do sleep 20; done
done
wait
for rock in $ROCKS; do
  [ -f DATA/rock_sst11/parts/$rock/structures.csv ] || { echo "ABORT: $rock part missing, see GEN_$rock.log"; exit 1; }
done

echo "########## merge  $(date -Is) ##########"
python scripts/stretch_rock.py --merge --out DATA/rock_sst11
echo "SST11 GEN DONE $(date -Is)"
