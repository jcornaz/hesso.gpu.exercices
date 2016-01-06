#include <iostream>

extern float computePIWithSlicing();

bool useSlicing();

bool useSlicing() {
  float piValue = computePIWithSlicing();

  std::cout << "PI = " << piValue << " (with slicing)" << std::endl;

  return abs(piValue - 3.141592653589793f) < 0.0001;
}
