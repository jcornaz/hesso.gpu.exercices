#ifndef JULIA_H_
#define JULIA_H_

#include "Fractale.h"

class Julia: public Fractale {
  public:
    Julia(double c1, double c2) {
      this->c1 = c1;
      this->c2 = c2;
    }

  	int checkSuit(double x, double y, int n) const {
      int k = 0;

      double a = x;
      double b = y;
      double aSquared = a * a;
      double bSquared = b * b;

      while (k <= n && aSquared + bSquared <= 4.0) {

        b = 2 * a * b + this->c2;
        a = aSquared - bSquared + this->c1;

        aSquared = a * a;
        bSquared = b * b;

        k++;
      }

      return k;
    }

    string getName() const {
      return "Julia";
    }

  private:
    double c1;
    double c2;
};

#endif
