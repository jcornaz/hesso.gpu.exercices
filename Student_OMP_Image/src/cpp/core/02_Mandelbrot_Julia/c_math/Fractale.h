#ifndef FRACTALE_H_
#define FRACTALE_H_

#include <cmath>

#include "ColorTools.h"
#include "cudaType.h"

class Fractale {
  public:

  	void colorXY(uchar4* ptrColor, double x, double y, int n) const {
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

  	virtual int checkSuit(double x, double y, int n) const = 0;

    virtual string getName() const = 0;
};

#endif
