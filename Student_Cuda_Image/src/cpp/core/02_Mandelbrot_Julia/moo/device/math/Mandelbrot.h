#ifndef MANDELBROT_H_
#define MANDELBROT_H_

#include "Fractale.h"

class Mandelbrot: public Fractale {

  public:

    __device__ void colorXY(uchar4* ptrColor, double x, double y, int n ) const {

      int k = this->checkSuit(x, y, n);

      float3 hsb;

      if (k > n) {
        hsb.x = 0;
        hsb.y = 0;
        hsb.z = 0;
      } else {
        hsb.x = ((float) k) / ((float) n);
        hsb.y = 1;
        hsb.z = 1;
      }

      ColorTools::HSB_TO_RVB(hsb, ptrColor);

      ptrColor->w = 255;
    }

  	__device__ int checkSuit(double x, double y, int n) const {
      int k = 0;

      double a = 0.0;
      double b = 0.0;
      double aPowed = 0.0;
      double bPowed = 0.0;

      while (k <= n) {
        aPowed = a * a;
        bPowed = b * b;
        if ((aPowed + bPowed) > 4.0) {  // This line fail with : "[CUDA ERROR] : Fractale_Cuda: an illegal memory access was encountered"
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
