#!/bin/bash
# The 2D descriptors recomputed on the VOID phase -- the pore space that
# actually carries the flow -- instead of the solid skeleton inherited from the
# elasticity code (filtration2d.py --phase, NOTES 11.8).
#
# Both ECP and PH must be recomputed here, unlike the bandwidth rerun: that one
# changed only how diagrams were imaged, this one changes the filtration
# itself, so nothing can be symlinked from DESC/aniso.
#
# PH imaging parameters are NOT the ones from the bandwidth fix, and copying
# them across would have reintroduced exactly the bug session 7 removed.
# Measured on 40 structures, persistence p99:
#
#            H0 birth   H0 pers   H1 birth   H1 pers
#   solid      0.96       1.25       0.94       0.11
#   void       0.73-0.85  0.31       0.85-0.95  1.25
#
# The two dimensions SWAP roles with the phase, which is what duality predicts:
# in the solid phase the long-lived class is H0 (components of the skeleton),
# in the void phase it is H1 (loops of pore around solid grains). The solid
# settings put H1 on p[0,0.10]; on void that would crush the informative
# dimension into a single pixel row. So each dimension gets a range covering
# its own p99 and a bandwidth at the geometric mean of the two pixel sizes:
#
#   H0  b[0,1.0] p[0,0.35]  res 20  -> px 0.050 / 0.0175  -> bw 0.030
#   H1  b[0,1.0] p[0,1.25]  res 20  -> px 0.050 / 0.0625  -> bw 0.056
#
# Filtrations go to $SCRATCH: 2.4 GiB per direction, and /home is quota-limited.
set -e
cd /home/rtopolnicki/flow/flow-anizotropy

SCRATCH="${SCRATCH:?set SCRATCH to a directory with ~3 GiB free}"
DESCROOT=${DESCROOT:-DESC/aniso_void}
WORKERS=${WORKERS:-24}
JLENV=${JLENV:-$HOME/direction-aware-tda-for-porous-materials}
RES=20

for TH in 0 45 90 135; do
  echo "=== direction ${TH} deg (void phase) ==="
  FILT="$SCRATCH/filt_void_a$TH"

  python scripts/filtration2d.py --dataset DATA/aniso --outputdir "$FILT" \
      --direction "$TH" --phase void \
      --radius 4 --wedge-radius 3 --wedge-height 6 --workers "$WORKERS"

  julia --project="$JLENV" --threads "$WORKERS" scripts/ecp2d.jl \
      "$FILT" "$DESCROOT/ecp/a$TH" --grid-res 8

  python scripts/ph2d.py --inputdir "$FILT" --outputdir "$DESCROOT/ph/a$TH" \
      --weight linear --resolution $RES $RES \
      --im_range_h0 0 1.0 0 0.35 --bandwidth_h0 0.030 \
      --im_range_h1 0 1.0 0 1.25 --bandwidth_h1 0.056 \
      --workers "$WORKERS"

  rm -rf "$FILT"
done

python scripts/collect_descriptors.py --dataset DATA/aniso \
    --descroot "$DESCROOT" --output "$DESCROOT/descriptors.csv"
echo "DONE"
