#ifndef NEWTON_MATH_H_
#define NEWTON_MATH_H_

#include <cmath>

#include "ColorTools.h"
#include "cudaType.h"

class NewtonMath {
  public:

  	void colorXY(uchar4* ptrColor, double x, double y, double t) const {
      double x;

      const double EPSILON = 0.0001;

      x = x - jx(x) * fy(x);
      y = y - jy(y) * fy(y);
    }
};

#endif
