#include "RipplingProvider.h"

Rippling* RipplingProvider::createMOO() {
  float dt = 1;

  int dw = 512;
  int dh = 512;

  return new Rippling(dw, dh, dt);
}

Image* RipplingProvider::createGL() {
 return new Image(createMOO());
}
