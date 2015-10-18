#include <iostream>
#include "Fractale.h"

using std::cout;
using std::endl;

Fractale::~Fractale() {
    // Nothing
}

void Fractale::colorXY(uchar4* ptrColor, float x, float y, int n) {
  int nmax = this->checkSuit(x, y, n);

  float3 hsb;

  if (nmax > n) {
    hsb.x = 0;
    hsb.y = 0;
    hsb.z = 0;
  } else {
    hsb.x = ((float) nmax) / ((float) n);
    hsb.y = 1;
    hsb.z = 1;
  }

  ColorTools::HSB_TO_RVB(hsb, ptrColor);

  ptrColor->w = 255;
}
