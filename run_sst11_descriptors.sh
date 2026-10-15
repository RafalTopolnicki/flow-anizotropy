#!/bin/bash
# Every descriptor for the 11 Sandstones multi-rock set (DATA/rock_sst11, 4400 cubes):
#
#   1-5. run_rock_descriptors.sh with DATASET=DATA/rock_sst11 ->
#          DESC/rock_sst11_cone/{baselines,descriptors}.csv     por_ tpc_ fab_ fabdir_ | ecp_ ph_
#          DESC/rock_sst11_throat/descriptors.csv               thr_
#          DESC/rock_sst11_throat_iso/descriptors.csv           thr_a1_0_0_ (iso)
#   6.   physbaselines3d.py -> DESC/rock_sst11_phys/physbaselines.csv     kc_ cp_
#
# Needs no labels: runs while the LBM is going. Resumable: rerun the same command;
# every stage skips finished work (per-chunk markers on SCRATCH, complete CSVs).
#
#   nohup bash run_sst11_descriptors.sh > SST11_DESC.log 2>&1 &
#
# Env: SCRATCH (default /tmp/$USER/sst11desc -- must be a LOCAL disk, not /home:
#      ~125k intermediate .npy files vs the /home 1M-inode quota; keep it until the
#      run is DONE or the cache is lost), WORKERS (default: all cores).
#
# Training afterwards (leave-one-rock-out):
#   --descriptors DESC/rock_sst11_cone/descriptors.csv DESC/rock_sst11_throat/descriptors.csv \
#                 DESC/rock_sst11_cone/baselines.csv DESC/rock_sst11_phys/physbaselines.csv \
#   --group-cv rock
#
# Imaging -- re-measured on rock_sst11, NOT copied from Berea (2026-10-15; 22 cubes,
# 2 per rock; directions [1,0,0] [1,1,0] [0,1,-1]; rule of NOTES 11.11: each
# dimension's range covers its own p99, bandwidth = geometric mean of the pixel sizes):
#
#   cone     H0 pers p99   H1 pers p99   H2 pers p99   H2 birth p1-p99
#   Berea      .42-.45       .96-1.00      .17-.55       .38-.99
#   sst11      .46-.48       .94-.97       .84-.90       .24-.99
#   -> cone H2 window p[0,.60] b[.3,1] would drop 2.5-4% of H2 bars: re-imaged to
#      b[.2,1.0] p[0,.95], bandwidth .087. H0/H1 windows already cover p99: KEPT.
#
#   throat   H0 pers p99 .32-.35 (window .40), H1 .35-.45 (.75), H2 .35 (.50): KEPT,
#            so thr_ columns are imaged as on Berea and the synthetic set.
#   iso      H0 pers p99 .357 at the Berea window edge .35 (1.1% beyond): widened to
#            p[0,.40], bandwidth .060. H1/H2 kept.
set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")"

export DATASET=DATA/rock_sst11
export PREFIX=DESC/rock_sst11
export SCRATCH=${SCRATCH:-/tmp/$USER/sst11desc}
export WORKERS=${WORKERS:-$(nproc)}

export C_RANGE_H2="0.2 1.0 0 0.95"; export C_BW_H2=0.087
export I_RANGE_H0="0 0.9 0 0.40";   export I_BW_H0=0.060
export IMAGING_NOTE="Imaging re-measured on rock_sst11 2026-10-15 (cone H2, iso H0 widened), see the header of run_sst11_descriptors.sh."

n=$(( $(wc -l < $DATASET/structures.csv) - 1 ))
[ "$n" -eq 4400 ] || { echo "ABORT: $DATASET/structures.csv has $n rows, expected 4400"; exit 1; }
mkdir -p "$SCRATCH"
echo "########## sst11 descriptors on $(hostname -s), $WORKERS workers, scratch $SCRATCH  $(date -Is) ##########"

# ---- 1-5. baselines, cone, throat, throat iso, collect ----------------------------
bash run_rock_descriptors.sh

# ---- 6. physics baselines (Kozeny-Carman + critical path) --------------------------
PHYS=DESC/rock_sst11_phys/physbaselines.csv
if [ -f "$PHYS" ] && [ $(( $(wc -l < "$PHYS") - 1 )) -eq 4400 ]; then
  echo "=== phys -- already complete, skipping ==="
else
  echo "=== phys  $(date -Is) ==="
  mkdir -p DESC/rock_sst11_phys
  python scripts/physbaselines3d.py --dataset $DATASET --output "$PHYS" --workers "$WORKERS"
fi

# ---- check: every CSV has all 4400 rows and no NaN ----------------------------------
python - <<'EOF'
import pandas as pd
for f in ["DESC/rock_sst11_cone/baselines.csv", "DESC/rock_sst11_cone/descriptors.csv",
          "DESC/rock_sst11_throat/descriptors.csv", "DESC/rock_sst11_throat_iso/descriptors.csv",
          "DESC/rock_sst11_phys/physbaselines.csv"]:
    d = pd.read_csv(f)
    num = d.select_dtypes("number")
    nan = int(num.isna().sum().sum())
    print(f"{f}: {len(d)} rows x {d.shape[1]} cols, NaN {nan}"
          + ("" if len(d) == 4400 and nan == 0 else "   <-- CHECK"))
EOF
echo "SST11 DESCRIPTORS DONE $(date -Is)"
