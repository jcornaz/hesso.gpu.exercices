#include "HeatTransfertProvider.h"
#include "HeatTransfertMOO.h"

Image* HeatTransfertProvider::createGL() {
  Animable_I* ptrAnimable = HeatTransfertProvider::createMOO();
  return new Image(ptrAnimable);
}

Animable_I* HeatTransfertProvider::createMOO() {
  return new HeatTransfertMOO(800, 800, NULL, NULL);
}
