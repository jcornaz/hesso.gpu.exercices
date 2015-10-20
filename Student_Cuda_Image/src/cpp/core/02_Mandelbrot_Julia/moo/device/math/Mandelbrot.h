#ifndef MANDELBROT_H_
#define MANDELBROT_H_

#include "Fractale.h"

class Mandelbrot: public Fractale {
  public:

  	__device__ int checkSuit(float x, float y, int n) const {
      int k = 0;

      float a = 0;
      float b = 0;
      float aPowed;
      float bPowed;

      while (k <= n) {
        aPowed = pow(a, 2);
        bPowed = pow(b, 2);
        if (aPowed + bPowed > 4) {
          break;
        } else {
          b = 2 * a * b + y;
          a = aPowed - bPowed + x;
          k++;
        }
      }

      return k;
    }
};

#endif
