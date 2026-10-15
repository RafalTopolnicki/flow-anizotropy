#!/bin/bash
# Train every model on every target set and collect one comparison table.
#
#   targets   K11 (k_xx), K22 (k_yy), K12 (k_off), and the full tensor
#   CatBoost  each feature group: TDA (ecp, ph, ecp+ph) and baselines
#             (porosity profile, TPC, fabric), plus the scalar-porosity null
#   CNN       convnext / resnet / densenet, straight from the image
#
# K12 is the symmetrised off-diagonal: K = K^T holds exactly for Stokes flow,
# so averaging k_xy and k_yx halves the noise on the component that needs it
# most.  full_tensor is a genuine joint model (CatBoost MultiRMSE; the CNN is
# multi-output natively), not three separate fits — so comparing it against the
# three single-target runs measures whether joint training helps.
#
# Usage:
#   bash run_training.sh                      # everything
#   TARGETS=k12 bash run_training.sh          # just the interesting one
#   SKIP_CNN=1 bash run_training.sh           # CatBoost only
#   LIMIT=500 EPOCHS=5 bash run_training.sh   # quick smoke run
set -euo pipefail

DATASET=${DATASET:-DATA/aniso}
LABELS=${LABELS:-$DATASET/permeability.csv}
DESC=${DESC:-DESC/aniso}
RESULTS=${RESULTS:-RESULTS}

TARGETS=${TARGETS:-"k11 k22 k12 full_tensor"}
CB_GROUPS=${CB_GROUPS:-"tda ecp ph baselines fab tpc por porosity"}
BACKBONES=${BACKBONES:-"convnext_tiny resnet50 densenet121"}

FOLDS=${FOLDS:-5}
SEED=${SEED:-0}
LIMIT=${LIMIT:-0}
STRATA=${STRATA:-}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-32}
AUGMENT=${AUGMENT:-roll}
SKIP_CNN=${SKIP_CNN:-0}
SKIP_CB=${SKIP_CB:-0}

[ -f "$LABELS" ] || { echo "no labels at $LABELS — has the LBM run finished?"; exit 1; }

EXTRA=""
[ "$LIMIT" != "0" ] && EXTRA="$EXTRA --limit $LIMIT"
[ -n "$STRATA" ] && EXTRA="$EXTRA --strata $STRATA"

echo "labels   $LABELS  ($(($(wc -l < "$LABELS") - 1)) rows)"
echo "targets  $TARGETS"
echo "results  $RESULTS"
echo

# --------------------------------------------------------------------------
for T in $TARGETS; do
  # the joint tensor is one multi-output model; the single components are not
  MULTI=""
  [ "$T" = "full_tensor" ] && MULTI="--multioutput"

  if [ "$SKIP_CB" != "1" ]; then
    echo "=== CatBoost | target $T ==="
    python scripts/train_catboost.py \
      --dataset "$DATASET" --labels "$LABELS" \
      --descriptors "$DESC/descriptors.csv" "$DESC/baselines.csv" \
      --groups $CB_GROUPS --targets "$T" $MULTI \
      --folds "$FOLDS" --seed "$SEED" $EXTRA \
      --output "$RESULTS/catboost/$T"
    echo
  fi

  if [ "$SKIP_CNN" != "1" ]; then
    for BB in $BACKBONES; do
      echo "=== CNN $BB | target $T ==="
      python scripts/train_cnn.py \
        --dataset "$DATASET" --labels "$LABELS" \
        --backbone "$BB" --targets "$T" \
        --augment "$AUGMENT" --epochs "$EPOCHS" --batch "$BATCH" \
        --folds "$FOLDS" --seed "$SEED" $EXTRA \
        --output "$RESULTS/cnn/${BB}_$T"
      echo
    done
  fi
done

# --------------------------------------------------------------------------
echo "=== summary ==="
python - "$RESULTS" <<'PY'
import glob, os, sys
import pandas as pd

root = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "*", "results.csv"))):
    d = pd.read_csv(f)
    d["run"] = os.path.basename(os.path.dirname(f))
    d["kind"] = os.path.basename(os.path.dirname(os.path.dirname(f)))
    rows.append(d)

if not rows:
    sys.exit("no results found")

r = pd.concat(rows, ignore_index=True)
r.to_csv(os.path.join(root, "summary.csv"), index=False)

if "mode" not in r.columns:
    r["mode"] = "single"
r["mode"] = r["mode"].fillna("single")
# keep `mode` in the index: a joint fit and a single fit of the same target are
# different experiments, and aggregating over them hides which one won
piv = (r.pivot_table(index=["kind", "group", "mode"], columns="target",
                     values="r2_mean", aggfunc="max")
        .round(4).sort_index())
print(piv.to_string())
print(f"\n-> {root}/summary.csv   ({len(r)} rows)")

if "k_off" in r.target.values:
    best = (r[r.target == "k_off"].sort_values("r2_mean", ascending=False)
              .head(6)[["kind", "group", "mode", "r2_mean", "r2_std"]])
    print("\nBest on k_off (K12) — the component the study turns on:")
    print(best.to_string(index=False))
PY
