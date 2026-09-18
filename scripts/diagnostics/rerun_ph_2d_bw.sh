#!/bin/bash
# Recompute the 2D persistence images with a CORRECTED BANDWIDTH (NOTES 11.5a).
#
# Two configurations are imaged from the same filtrations, because the
# filtration pass dominates the cost and imaging is nearly free:
#
#   A  DESC/aniso_ph_bw125      bandwidth 0.125 for both dimensions -- the
#                               configuration named in the resume block.
#   B  DESC/aniso_ph_bw125_h1s  bandwidth 0.125 for H0, 0.0158 for H1.
#
# B exists because the per-dimension im_range and the shared bandwidth are
# inconsistent: with H1 on p[0,0.10] at resolution 20 a persistence pixel is
# 0.005, so bandwidth 0.125 is 25 px -- the very degeneracy being fixed, left
# in place on H1. 0.0158 is the geometric mean of the H1 birth and persistence
# pixel sizes (0.05, 0.005); the grid is intrinsically anisotropic and gudhi's
# kernel is not, so some compromise is unavoidable. Measured rank on 60
# structures: 4 (A) vs 16 (B).
#
# Filtrations go to /tmp: 2.4 GiB per direction, and /home is quota-limited.
# ECP is unaffected (it reads the orientation channel, PH the wedge channel),
# so each descroot symlinks the existing one rather than recomputing it.
set -e
cd /home/rtopolnicki/flow/flow-anizotropy

SCRATCH="${SCRATCH:?set SCRATCH to a directory with ~3 GiB free}"
A=DESC/aniso_ph_bw125
B=DESC/aniso_ph_bw125_h1s
RES=20

for TH in 0 45 90 135; do
  echo "=== direction ${TH} deg ==="
  FILT="$SCRATCH/filt_a$TH"
  python scripts/filtration2d.py --dataset DATA/aniso --outputdir "$FILT" \
      --direction "$TH" --radius 4 --wedge-radius 3 --wedge-height 6 --workers 24

  python scripts/ph2d.py --inputdir "$FILT" --outputdir "$A/ph/a$TH" \
      --weight linear --resolution $RES $RES \
      --bandwidth 0.125 \
      --im_range_h0 0 1.25 0 1.25 --im_range_h1 0 1.0 0 0.10 --workers 24

  python scripts/ph2d.py --inputdir "$FILT" --outputdir "$B/ph/a$TH" \
      --weight linear --resolution $RES $RES \
      --bandwidth 0.125 --bandwidth_h1 0.0158 \
      --im_range_h0 0 1.25 0 1.25 --im_range_h1 0 1.0 0 0.10 --workers 24

  rm -rf "$FILT"
done

for D in "$A" "$B"; do
  [ -e "$D/ecp" ] || ln -s ../aniso/ecp "$D/ecp"
  python scripts/collect_descriptors.py --dataset DATA/aniso \
      --descroot "$D" --output "$D/descriptors.csv"
done
echo "DONE"
