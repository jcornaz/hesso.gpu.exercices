#ifndef JULIA_H_
#define JULIA_H_

#include "Fractale.h"

class Julia: public Fractale {
  public:
    Julia(float c1, float c2) {
      this->c1 = c1;
      this->c2 = c2;
    }

  	int checkSuit(float x, float y, int n) const {
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

    string getName() const {
      return "Julia";
    }

  private:
    float c1;
    float c2;
};

#endif
