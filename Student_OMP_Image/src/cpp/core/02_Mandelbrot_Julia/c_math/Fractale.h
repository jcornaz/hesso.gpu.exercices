#ifndef FRACTALE_H_
#define FRACTALE_H_

#include <cmath>

#include "ColorTools.h"
#include "cudaType.h"

class Fractale {
  public:

  	void colorXY(uchar4* ptrColor, float x, float y, int n) {
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

  	virtual int checkSuit(float x, float y, int n)=0;
};

#endif
