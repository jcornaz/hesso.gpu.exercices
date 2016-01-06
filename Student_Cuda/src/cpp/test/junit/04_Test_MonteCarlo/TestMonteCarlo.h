#ifndef TEST_MONTE_CARLO_H
#define TEST_MONTE_CARLO_H

#include "cpptest.h"

using Test::Suite;

class TestMonteCarlo: public Suite {
  public:
    TestMonteCarlo(int deviceId);

  private:
    int deviceId;

    void testPI();
    void testPIOnMultiGPU();
};

#endif
