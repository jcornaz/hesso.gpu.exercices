#ifndef JULIA_H_
#define JULIA_H_

#include "Fractale.h"

class Julia: public Fractale {
  public:
    Julia(float c1, float c2) {
      this->c1 = c1;
      this->c2 = c2;
    }

  	int checkSuit(float x, float y, int n) {

      float z = 0;
      int k = 0;

      float a = x;
      float b = y;
      float aPowed;
      float bPowed;

      while (k <= n) {
        aPowed = pow(a, 2);
        bPowed = pow(b, 2);
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

  private:
    float c1;
    float c2;
};

#endif
