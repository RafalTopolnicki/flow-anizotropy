g++ *.cpp -O3 -o lbm2d-256x256_force \
  $( $HOME/local-im6/bin/Magick++-config --cppflags --cxxflags ) \
  -I$HOME/local/include \
  -L$HOME/local/lib \
  -L$HOME/local-im6/lib \
  -Wl,-rpath,$HOME/local/lib:$HOME/local-im6/lib \
  -lGLEW -lGL -lGLU -lglut \
  $( $HOME/local-im6/bin/Magick++-config --ldflags --libs )
