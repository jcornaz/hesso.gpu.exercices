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

      this->A1 = 1.0f;
      this->A2 = 0.0f;

      this->B1 = -0.5f;
      this->B2 = sqrtf(3) / 2.0f;

      this->C1 = this->B1;
      this->C2 = - this->B2;
    }

    void colorXY(uchar4 *ptrColor, float x1, float x2, int n) {
      ptrColor->x = 0;
      ptrColor->y = 0;
      ptrColor->z = 0;
      ptrColor->w = 255;

      int k = this->checkConvergency(&x1, &x2, n);
      if (k <= n) {
        float distA = magnitude(x1 - this->A1, x2 - this->A2);
        float distB = magnitude(x1 - this->B1, x2 - this->B2);
        float distC = magnitude(x1 - this->C1, x2 - this->C2);

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
    float A1, A2, B1, B2, C1, C2;

    int checkConvergency(float* x, float* y, int n) {
      // assert n >= 0

      int k = 0;
      bool isConvergent = false;

      do {

        float a = this->d1f1(*x, *y);
        float b = this->d1f2(*x, *y);
        float c = this->d2f1(*x, *y);
        float d = this->d2f2(*x, *y);

        float sa = d / (a * d - b * c);
        float sb = -c / (a * d - b * c);
        float sc = -b / (a * d - b * c);
        float sd = a / (a * d - b * c);

        float nextX = *x - (sa * this->f1(*x, *y) + sc * this->f2(*x, *y));
        float nextY = *y - (sb * this->f1(*x, *y) + sd * this->f2(*x, *y));

        isConvergent = (magnitude(*x - this->A1, *y - this->A2) / magnitude(this->A1, this->A2)) < this->epsilon;
        isConvergent = isConvergent || (magnitude(*x - this->B1, *y - this->B2) / magnitude(this->B1, this->B2)) < this->epsilon;
        isConvergent = isConvergent || (magnitude(*x - this->C1, *y - this->C2) / magnitude(this->C1, this->C2)) < this->epsilon;

        *x = nextX;
        *y = nextY;

        k++;
      } while (!isConvergent && k <= n);

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
