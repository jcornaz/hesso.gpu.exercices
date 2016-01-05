#ifndef NEWTON_MATH_H_
#define NEWTON_MATH_H_

#include <math.h>
#include <cfloat>
#include "ColorTools.h"

class NewtonMath {

  public:
    NewtonMath(float epsilon, int colorFactor) {
      this->epsilon = epsilon;
      this->colorFactor = colorFactor;

      this->XA1 = 1.0f;
      this->XA2 = 0.0f;

      this->XB1 = -0.5f;
      this->XB2 = sqrtf(3) / 2.0f;

      this->XC1 = this->XB1;
      this->XC2 = - this->XB2;
    }

    void colorXY(uchar4 *ptrColor, float x1, float x2, int n) {
      ptrColor->x = 0;
      ptrColor->y = 0;
      ptrColor->z = 0;
      ptrColor->w = 255;

      int k = this->checkConvergency(&x1, &x2, n);
      if (k <= n) {
        float distA = magnitude(x1 - this->XA1, x2 - this->XA2);
        float distB = magnitude(x1 - this->XB1, x2 - this->XB2);
        float distC = magnitude(x1 - this->XC1, x2 - this->XC2);

        float color = 255 - ((k * this->colorFactor) % 255);

        if (distA < distB && distA < distC) {
          ptrColor->x = color;
        } else if (distB < distC) {
          ptrColor->y = color;
        } else {
          ptrColor->z = color;
        }
      }
    }

  private:
    int colorFactor;
    float epsilon;
    float XA1, XA2, XB1, XB2, XC1, XC2;

    int checkConvergency(float* x1, float* x2, int n) {
      int k = 0;

      while (k <= n) {

        float a = this->d1f1(*x1, *x2);
        float b = this->d1f2(*x1, *x2);
        float c = this->d2f1(*x1, *x2);
        float d = this->d2f2(*x1, *x2);

        float jacobianInverse = 1.0f / (a * d - b * c);

        float sa = jacobianInverse * d;
        float sb = -jacobianInverse * c;
        float sc = -jacobianInverse * b;
        float sd = jacobianInverse * a;

        float nextX1 = *x1 - (sa * this->f1(*x1, *x2) + sc * this->f2(*x1, *x2));
        float nextX2 = *x2 - (sb * this->f1(*x1, *x2) + sd * this->f2(*x1, *x2));

        if (
          (magnitude(*x1 - this->XA1, *x2 - this->XA2) / magnitude(this->XA1, this->XA2)) < this->epsilon ||
          (magnitude(*x1 - this->XB1, *x2 - this->XB2) / magnitude(this->XB1, this->XB2)) < this->epsilon ||
          (magnitude(*x1 - this->XC1, *x2 - this->XC2) / magnitude(this->XC1, this->XC2)) < this->epsilon
        ) {
          break;
        } else {
          *x1 = nextX1;
          *x2 = nextX2;
          k++;
        }
      }

      return k;
    }

    float magnitude(float x, float y) {
        return sqrtf(x * x + y * y);
    }

    float f1(float x, float y) {
        return x * x * x - 3.0f * x * y * y - 1.0f;
    }

    float f2(float x, float y) {
        return y * y * y - 3.0f * x * x * y;
    }

    float d1f1(float x, float y) {
        return 3.0f * (x * x - y * y);
    }

    float d2f1(float x, float y) {
        return -6.0f * x * y;
    }

    float d1f2(float x, float y) {
        return d2f1(x, y);
    }

    float d2f2(float x, float y) {
        return 3.0f * (y * y - x * x);
    }
};

#endif
