#include "TestMonteCarlo.h"

extern bool useMonteCarlo();
extern bool useMonteCarloMultiGPU();

TestMonteCarlo::TestMonteCarlo(int deviceId) {
  this->deviceId=deviceId;

  TEST_ADD(TestMonteCarlo::testPI);
  TEST_ADD(TestMonteCarlo::testPIOnMultiGPU);
}

void TestMonteCarlo::testPI() {
  TEST_ASSERT(useMonteCarlo());
}

void TestMonteCarlo::testPIOnMultiGPU() {
  TEST_ASSERT(useMonteCarloMultiGPU());
}
