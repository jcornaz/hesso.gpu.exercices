#ifndef MANDELBROT_H_
#define MANDELBROT_H_

#include "Fractale.h"
#include <cmath>

class Mandelbrot: public Fractale {

  public:
  	__device__ int checkSuit(double x, double y, int n) const {

      int k = 0;

      double a = 0.0;
      double b = 0.0;
      double aSquared = 0.0;
      double bSquared = 0.0;

      while (k <= n && aSquared + bSquared <= 4.0) {

        b = 2 * a * b + y;
        a = aSquared - bSquared + x;

        aSquared = a * a;
        bSquared = b * b;

        k++;
      }

      return k;
    }
};

#endif
