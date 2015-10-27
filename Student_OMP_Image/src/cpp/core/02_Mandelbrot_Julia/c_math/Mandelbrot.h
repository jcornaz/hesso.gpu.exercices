#ifndef MANDELBROT_H_
#define MANDELBROT_H_

#include "Fractale.h"

class Mandelbrot: public Fractale {

  public:

  	int checkSuit(float x, float y, int n) const {
      int k = 0;

      float a = 0;
      float b = 0;
      float aPowed;
      float bPowed;

      while (k <= n) {
        aPowed = a * a;
        bPowed = b * b;
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

    string getName() const {
      return "Mandelbrot";
    }
};

#endif
