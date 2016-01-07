#include "HeatTransfertProvider.h"
#include "HeatTransfert.h"
#include "IndiceTools.h"

Image* HeatTransfertProvider::createGL() {
  Animable_I* ptrAnimable = HeatTransfertProvider::createMOO();
  return new Image(ptrAnimable);
}

Animable_I* HeatTransfertProvider::createMOO() {
  unsigned int w = 512;
  unsigned int h = 512;
  unsigned int WH = w*h;

  float imageInit[WH];
  float imageHeater[WH];

  for (int s = 0 ; s < WH ; s++) {
    imageInit[s] = 0.0;

    int i, j;
    IndiceTools::toIJ(s, w, &i, &j);

    if (i >= 150 && i < 250 && j >= 150 && j < 250) {
      imageHeater[s] = 1.0;
    } else if (
      (i >= 90 && i < 98 && j >= 90 && j < 98) ||
      (i >= 90 && i < 98 && j >= 302 && j < 310) ||
      (i >= 302 && i < 310 && j >= 90 && j < 98) ||
      (i >= 302 && i < 310 && j >= 302 && j < 310) ||
      (i >= 302 && i < 310 && j >= 302 && j < 310) ||
      (i >= 302 && i < 310 && j >= 302 && j < 310)
    ) {
      imageHeater[s] = 0.2;
    } else {
      imageHeater[s] = 0.0;
    }
  }

  return new HeatTransfert(w, h, imageInit, imageHeater, 0.25);
}
