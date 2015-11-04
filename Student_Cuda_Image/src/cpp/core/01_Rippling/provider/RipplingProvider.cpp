#include "RipplingProvider.h"

Rippling* RipplingProvider::createMOO() {
  float dt = 1;

  int dw = 16 * 64;
  int dh = 16 * 64;

  return new Rippling(dw, dh, dt);
}

Image* RipplingProvider::createGL(void) {
 return new Image(createMOO());
}
