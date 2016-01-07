#include <iostream>

extern float computePIWithMonteCarloSingleGPU();
extern float computePIWithMonteCarloMultiGPU();

bool useMonteCarlo();
bool useMonteCarloMultiGPU();

bool useMonteCarlo() {
  float piValue = computePIWithMonteCarloSingleGPU();
  std::cout << "PI = " << piValue << " (with Monte Carlo single GPU)" << std::endl;
  return abs(piValue - 3.141592653589793f) < 0.001;
}

bool useMonteCarloMultiGPU() {
  float piValue = computePIWithMonteCarloMultiGPU();
  std::cout << "PI = " << piValue << " (with Monte Carlo multi GPU)" << std::endl;
  return abs(piValue - 3.141592653589793f) < 0.001;
}
