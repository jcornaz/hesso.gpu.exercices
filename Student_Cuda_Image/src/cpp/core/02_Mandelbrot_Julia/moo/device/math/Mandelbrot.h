#ifndef MANDELBROT_H_
#define MANDELBROT_H_

#include "Fractale.h"
#include <cmath>

class Mandelbrot: public Fractale {

  public:

    __device__ void colorXY(uchar4* ptrColor, double x, double y, int n ) const {
      int nmax = this->checkSuit(x, y, n);

      float3 hsb;

      if (nmax > n) {
        hsb.x = 0;
        hsb.y = 0;
        hsb.z = 0;
      } else {
        hsb.x = ((double) nmax) / ((double) n);
        hsb.y = 1;
        hsb.z = 1;
      }

      ColorTools::HSB_TO_RVB(hsb, ptrColor);

      ptrColor->w = 255;
    }

  private:
  	__device__ int checkSuit(double x, double y, int n) const {

      int k = 0;

      double a = 0.0;
      double b = 0.0;
      double aSquared = 0.0;
      double bSquared = 0.0;

      while (k <= n) { // TODO && aSquared + bSquared <= 4.0

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
