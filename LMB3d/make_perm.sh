#!/bin/bash
# Headless build.  No external dependencies: the .raw structure format means
# there is no image library to link against, unlike the 2D solver's Magick++.
set -e
g++ -O3 -march=native -ffast-math -o lbm3d-perm lbm3d-perm.cpp
echo "built $(pwd)/lbm3d-perm"
