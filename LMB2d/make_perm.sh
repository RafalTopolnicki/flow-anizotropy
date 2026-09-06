#!/bin/bash
# Headless build of the permeability-tensor solver.
# Only main.cpp + lbm.cpp are needed; particles.cpp is GL-only visualisation
# and is deliberately excluded so the binary has no OpenGL dependency.
set -e
g++ main.cpp lbm.cpp -O3 -o lbm2d-perm \
  $( $HOME/local-im6/bin/Magick++-config --cppflags --cxxflags ) \
  -L$HOME/local-im6/lib \
  -Wl,-rpath,$HOME/local-im6/lib \
  $( $HOME/local-im6/bin/Magick++-config --ldflags --libs )
