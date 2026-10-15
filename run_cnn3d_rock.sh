#!/bin/bash
# All CNN experiments on the real-rock 3D set (stretched Berea, DATA/rock_berea,
# n=1000), on the GPU node:
#
#   1. single-target densenet121, AUGMENT=roll, one run per target:
#        K_xx k_xy k_xz k_yz
#      Same settings as the synthetic RESULTS/aniso3d/cnn3d/densenet121_<T>
#      runs (5 folds, 30 epochs, batch 8, seed 0), so the rows compare directly.
#   2. joint off-diagonal run with the 48-element octahedral group,
#      AUGMENT=roll+oct, targets k_xy k_xz k_yz together. The group can send
#      k_xy to +/-k_xz, so it needs the complete off-diagonal family and cannot
#      be used on the single-target runs (see run_cnn3d.sh).
#
# Resumable: a run whose metrics.json exists is skipped, so rerun after an
# interruption (a half-finished run restarts from its first fold).
#
#   nohup ./run_cnn3d_rock.sh > CNN_ROCK.log 2>&1 &
#
# Smoke test (a few minutes, writes elsewhere):
#   RESULTS=/tmp/cnn_smoke EPOCHS=1 LIMIT=100 ./run_cnn3d_rock.sh
#
# One GPU job at a time on this card.
set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")"

DATASET=${DATASET:-DATA/rock_berea}
RESULTS=${RESULTS:-RESULTS/rock_berea}
TARGETS=${TARGETS:-"K_xx k_xy k_xz k_yz"}
BACKBONE=${BACKBONE:-densenet121}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-8}
FOLDS=${FOLDS:-5}
SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda}
LIMIT=${LIMIT:-0}
OCT=${OCT:-1}            # 0 skips the joint roll+oct run

LABELS=$DATASET/permeability.csv
OUT=$RESULTS/cnn3d
EXTRA=""; [ "$LIMIT" != "0" ] && EXTRA="--limit $LIMIT"

# Labels must be the complete, merged set: a partial table would silently
# train on fewer cubes than the descriptor models.
python - "$LABELS" "$LIMIT" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); lim = int(sys.argv[2])
conv = int(d["conv_all"].sum()) if "conv_all" in d else len(d)
print(f"labels   {sys.argv[1]}  {len(d)} rows, {conv} converged")
if lim == 0 and (len(d) != 1000 or conv != len(d)):
    sys.exit("ABORT: expected 1000 converged rows -- run "
             "solve_permeability3d.py --merge-only first")
EOF
python -c "import torch; assert torch.cuda.is_available() or '$DEVICE' != 'cuda', 'no CUDA device'"
echo "backbone $BACKBONE   epochs $EPOCHS   batch $BATCH   folds $FOLDS   seed $SEED"
echo "results  $OUT"
mkdir -p "$OUT"

run () {   # run <output dir> <augment> <targets...>
  local DIR=$1 AUG=$2; shift 2
  if [ -f "$DIR/metrics.json" ]; then
    echo "=== $(basename "$DIR") -- done, skipping ==="; return 0
  fi
  echo "=== $(basename "$DIR")  augment=$AUG  targets=$*  $(date -Is) ==="
  python scripts/train_cnn3d.py \
    --dataset "$DATASET" --labels "$LABELS" \
    --backbone "$BACKBONE" --targets "$@" \
    --augment "$AUG" --epochs "$EPOCHS" --batch "$BATCH" \
    --folds "$FOLDS" --seed "$SEED" --device "$DEVICE" $EXTRA \
    --output "$DIR"
  echo "=== $(basename "$DIR") finished  $(date -Is) ==="
}

for T in $TARGETS; do
  run "$OUT/${BACKBONE}_$T" roll "$T"
done
if [ "$OCT" = "1" ]; then
  run "$OUT/${BACKBONE}_offdiag_oct" roll+oct k_xy k_xz k_yz
fi

# Summary: mean +/- sd of R^2 over folds, per run and target.
python - "$OUT" <<'EOF'
import sys, os, json, glob, numpy as np
print(f"\n{'run':34s} {'target':6s}  R2 mean   sd     MAE mean")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "metrics.json"))):
    m = json.load(open(f))
    for e in m.get("experiments", []):
        r2 = [x["r2"] for x in e["folds"]]; mae = [x["mae"] for x in e["folds"]]
        print(f"{os.path.basename(os.path.dirname(f)):34s} {e['target']:6s}  "
              f"{np.mean(r2):.4f}  {np.std(r2):.4f}  {np.mean(mae):.4f}")
EOF
echo "CNN ROCK DONE  $(date -Is)"
