#ifndef MANDELBROT_H_
#define MANDELBROT_H_

#include "Fractale.h"

class Mandelbrot: public Fractale {

  public:

  	__device__ void colorXY(uchar4* ptrColor, int nmax, int n ) const {

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

  	__device__ void checkSuit(float x, float y, int n, int* nmax) const {
      int k = 0;

      double a = 0;
      double b = 0;
      double aPowed;
      double bPowed;

      const double T = 4.0;

      while (k <= n) {
        aPowed = pow(a, 2.0);
        bPowed = pow(b, 2.0);
        if ((aPowed + bPowed) > T) {
          break;
        } else {
          b = 2.0 * a * b + y;
          a = aPowed - bPowed + x;
          k++;
        }
      }

      *nmax = k;
    }
};

#endif
