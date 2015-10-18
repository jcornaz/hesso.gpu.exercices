#include "Mandelbrot.h"
#include <cmath>


Mandelbrot::~Mandelbrot() {
  // NOthing
}

int Mandelbrot::checkSuit(float x, float y, int n) {

  float z = 0;
  int s = 0;

  while (s < n) {
    z = pow(z, 2) + (x + s * y);

    if (this->isDivergent(z))
      break;

    s++;
  }

  return s;
}

bool Mandelbrot::isDivergent(float z) {
  return std::abs(z) > 2.0;
}
