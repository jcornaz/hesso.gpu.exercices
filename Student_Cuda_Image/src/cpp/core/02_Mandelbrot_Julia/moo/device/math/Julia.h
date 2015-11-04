#ifndef JULIA_H_
#define JULIA_H_

#include "Fractale.h"

class Julia: public Fractale {
  
  public:

    __device__ Julia(float c1, float c2) {
      this->c1 = c1;
      this->c2 = c2;
    }

  	__device__ void colorXY(uchar4* ptrColor, float x, float y, int n) const {
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

  	__device__ int checkSuit(float x, float y, int n) const {
      int k = 0;

      float a = x;
      float b = y;
      float aPowed;
      float bPowed;

      while (k <= n) {
        aPowed = a * a;
        bPowed = b * b;
        if (aPowed + bPowed > 4) {
          break;
        } else {
          b = 2 * a * b + this->c2;
          a = aPowed - bPowed + this->c1;
          k++;
        }
      }

      return k;
    }

    float c1;
    float c2;
};

#endif
