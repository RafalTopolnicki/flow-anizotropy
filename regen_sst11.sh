#!/bin/bash
# Regenerate three parts of DATA/rock_sst11 (2026-10-15): BSG died silently in the first
# pass (likely OOM: 6 rock SDFs at once), BanderaGray g 3->4 and Leopard g 6->7 by the
# user's override (DATA/rock_sst11_grow.json). Ids as in run_sst11_generate.sh. Then merge.
set -u
cd "$(dirname "$(readlink -f "$0")")"
for spec in BanderaGray:400 BSG:1600 Leopard:3600; do
  rock=${spec%%:*}; off=${spec##*:}
  g=$(python -c "import json; print(json.load(open('DATA/rock_sst11_grow.json'))['rocks']['$rock']['grow'])")
  echo "$rock g=$g offset $off"
  python -u scripts/stretch_rock.py --generate --rock $rock --n 400 --scale 3 --grow $g \
      --id-offset $off --jobs 12 --out DATA/rock_sst11/parts/$rock > GEN_$rock.log 2>&1 &
done
wait
for rock in BanderaBrown BanderaGray Bentheimer Berea BSG BUG BB CastleGate Kirby Leopard Parker; do
  [ -f DATA/rock_sst11/parts/$rock/structures.csv ] || { echo "ABORT: $rock part missing"; exit 1; }
done
python scripts/stretch_rock.py --merge --out DATA/rock_sst11
echo "SST11 REGEN DONE $(date -Is)"
