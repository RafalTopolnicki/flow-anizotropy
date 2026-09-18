#!/bin/bash
# Recompute the 2D persistence images with a persistence weight.
# ECP is unaffected (it reads the orientation channel, PH reads the wedge
# channel), so DESC/aniso_phw/ecp is a symlink to the existing one rather than
# a recomputation -- collect_descriptors then builds a complete table.
set -e
cd /home/rtopolnicki/flow/flow-anizotropy
for TH in 0 45 90 135; do
  echo "=== direction ${TH} deg ==="
  FILT=DESC/aniso_phw/filt/a$TH
  python scripts/filtration2d.py --dataset DATA/aniso --outputdir "$FILT" \
      --direction "$TH" --radius 4 --wedge-radius 3 --wedge-height 6 --workers 24
  python scripts/ph2d.py --inputdir "$FILT" --outputdir DESC/aniso_phw/ph/a$TH \
      --weight linear --resolution 10 10 --workers 24
  rm -rf "$FILT"
done
python scripts/collect_descriptors.py --dataset DATA/aniso \
    --descroot DESC/aniso_phw --output DESC/aniso_phw/descriptors.csv
