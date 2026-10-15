#!/bin/bash
# Join the two LBM parts of DATA/rock_sst11 into permeability.csv.
# Run once BOTH `run_lbm_sst11.sh 0` and `run_lbm_sst11.sh 1` have finished.
#
#   shared disk (both machines wrote into DATA/rock_sst11/perm):
#       bash merge_lbm_sst11.sh
#   separate disks: copy the other machine's perm/ directory here first, e.g.
#       rsync -a otherhost:<repo>/DATA/rock_sst11/perm/ /tmp/perm_part1/
#       bash merge_lbm_sst11.sh /tmp/perm_part1
#     (any number of directories; only finished per-sample CSVs are copied in,
#      nothing already here is overwritten)
#
# Then: audit (all 4400 complete?), --merge-only -> DATA/rock_sst11/permeability.csv,
# and a per-rock summary.
set -u
cd "$(dirname "$(readlink -f "$0")")"
DATASET=DATA/rock_sst11
PY=${PY:-python}

for src in "$@"; do
  [ -d "$src" ] || { echo "ABORT: $src is not a directory"; exit 1; }
  k=$(find "$src" -maxdepth 1 -name '[0-9][0-9][0-9][0-9][0-9][0-9].csv' | wc -l)
  echo "copying $k per-sample CSVs from $src"
  find "$src" -maxdepth 1 -name '[0-9][0-9][0-9][0-9][0-9][0-9].csv' \
      -exec cp -n -t $DATASET/perm/ {} +
done

# audit the whole set; incomplete ones are reported, not silently dropped
$PY scripts/solve_permeability3d.py --dataset $DATASET --verify

$PY scripts/solve_permeability3d.py --dataset $DATASET --merge-only || exit 1

$PY - <<'EOF'
import pandas as pd
p = pd.read_csv("DATA/rock_sst11/permeability.csv")
print(f"\n{len(p)}/4400 labelled, {int(p.conv_all.sum())} converged in all three directions")
if len(p) < 4400:
    print("WARNING: labels missing -- rerun the part(s) that are short "
          "(bash lbm_sst11_status.sh shows which), then merge again")
p["koff"] = p[["koff_over_trace_xy", "koff_over_trace_xz", "koff_over_trace_yz"]].abs().max(axis=1)
t = p.groupby("rock").agg(n=("sample_id", "size"), conv=("conv_all", "sum"),
                          por=("porosity", "mean"), k_mean=("k_mean", "median"),
                          k1_k3=("k_ratio", "median"), koff_trace=("koff", "median"),
                          recip=("recip_resid", "median"))
print(t.to_string(float_format=lambda v: f"{v:.3g}"))
EOF
echo "MERGE SST11 DONE $(date -Is)"
