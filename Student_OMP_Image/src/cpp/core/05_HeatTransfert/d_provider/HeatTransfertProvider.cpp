#include "HeatTransfertProvider.h"
#include "HeatTransfertMOO.h"
#include "IndiceTools.h"

Image* HeatTransfertProvider::createGL() {
  Animable_I* ptrAnimable = HeatTransfertProvider::createMOO();
  return new Image(ptrAnimable);
}

Animable_I* HeatTransfertProvider::createMOO() {
  unsigned int w = 800;
  unsigned int h = 800;
  unsigned int WH = w*h;

  float imageInit[WH];
  float imageHeater[WH];

  for (int s = 0 ; s < WH ; s++) {
    imageInit[s] = 0.0;

    int i, j;
    IndiceTools::toIJ(s, w, &i, &j);

    if (i >= 300 && i < 500 && j >= 300 && j < 500) {
      imageHeater[s] = 1.0;
    } else if (
      (i >= 179 && i < 195 && j >= 179 && j < 195) ||
      (i >= 179 && i < 195 && j >= 605 && j < 621) ||
      (i >= 605 && i < 621 && j >= 179 && j < 195) ||
      (i >= 605 && i < 621 && j >= 605 && j < 621) ||
      (i >= 605 && i < 621 && j >= 605 && j < 621) ||
      (i >= 605 && i < 621 && j >= 605 && j < 621)
    ) {
      imageHeater[s] = 0.2;
    } else {
      imageHeater[s] = 0.0;
    }
  }

  return new HeatTransfertMOO(w, h, imageInit, imageHeater, 0.1);
}
