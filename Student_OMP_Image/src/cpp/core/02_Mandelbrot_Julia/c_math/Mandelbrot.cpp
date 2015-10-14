#include "Mandelbrot.h"
#include <math.h>


Mandelbrot::~Mandelbrot()
    {
    // NOthing
    }

int Mandelbrot::checkSuit(float x, float y, int n) {
  unsigned int nmax = 0;

  float z = 0;
  int s = 0;

  if (this->isDivergent(0)) {
    while (s < n && nmax > n) {
      z = pow(z, 2) + (x + s * y);

      if (this->isDivergent(z))
        break;

      s++;
    }
  }

  return s;
}

bool Mandelbrot::isDivergent(float z) {
  return (z >= 0 ? z : -z) > 2.0;
}
