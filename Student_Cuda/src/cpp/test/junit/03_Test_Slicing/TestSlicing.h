#ifndef TEST_SLICING_H
#define TEST_SLICING_H

#include "cpptest.h"

using Test::Suite;

class TestSlicing: public Suite {
  public:
    TestSlicing(int deviceId);

  private:
    int deviceId;

    void testPI();
};

#endif
