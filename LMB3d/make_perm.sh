#!/bin/bash
# Headless build.  No external dependencies: the .raw structure format means
# there is no image library to link against, unlike the 2D solver's Magick++.
#
# -march=broadwell, not -march=native: the solves run on the Xeon E5-2690 v4
# (Broadwell: AVX2 + FMA, no AVX-512) nodes, but the binary sits on the shared
# disk and may be built from any node.  A native build on an AVX-512 machine
# would die there with an illegal instruction.
#
# Safe to run while solves are in progress: the binary is built under a temp
# name and moved into place with an atomic rename.  Running solves keep the old
# file (they hold its inode); every solve started afterwards gets the new one.
set -e
cd "$(dirname "$(readlink -f "$0")")"
g++ -O3 -march=broadwell -ffast-math -o lbm3d-perm.new lbm3d-perm.cpp
[ -f lbm3d-perm ] && cp -p lbm3d-perm lbm3d-perm.prev
mv -f lbm3d-perm.new lbm3d-perm
echo "built $(pwd)/lbm3d-perm (previous kept as lbm3d-perm.prev)"
