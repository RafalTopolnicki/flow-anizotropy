#!/bin/bash
# Progress of run_lbm_sst11.sh (both parts): solved cubes per rock and per part, rate, ETA.
# Safe to run any time, from any machine that sees the same disk.
#   bash lbm_sst11_status.sh
cd "$(dirname "$(readlink -f "$0")")"
${PY:-python} - <<'EOF'
import os, time
import pandas as pd

ds = "DATA/rock_sst11"
s = pd.read_csv(f"{ds}/structures.csv")
perm = f"{ds}/perm"
files = {f: os.path.getmtime(os.path.join(perm, f))
         for f in (os.listdir(perm) if os.path.isdir(perm) else []) if f.endswith(".csv")}
tmp = sum(".tmp." in f for f in (os.listdir(perm) if os.path.isdir(perm) else []))
s["done"] = [f"{i:06d}.csv" in files for i in s.sample_id]

print(s.groupby("rock").done.agg(["sum", "size"]).rename(
    columns={"sum": "solved", "size": "of"}).T.to_string())
n, tot = int(s.done.sum()), len(s)
print(f"\ntotal {n}/{tot} ({100 * n / tot:.1f}%), in progress (.tmp) {tmp}")
for part in (0, 1):
    m = s.sample_id % 2 == part
    print(f"  part {part}: {int(s.done[m].sum())}/{int(m.sum())}")

now = time.time()
for h in (1, 6):
    k = sum(now - t < 3600 * h for t in files.values())
    if k:
        rate = k / h
        print(f"last {h} h: {k} solved = {rate:.1f}/h -> ETA {(tot - n) / rate:.1f} h")
if not files:
    print("nothing solved yet")
EOF
for f in LBM_SST11_part0.log LBM_SST11_part1.log; do
  [ -f "$f" ] && echo "$f: $(tail -c 300 "$f" | tr '\r' '\n' | grep -v '^$' | tail -n 1)"
done
