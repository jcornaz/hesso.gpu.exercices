#include "../02_Test_Vector/TestVector.h"

#include "Device.h"
#include "TestSlicing.h"

extern bool useSaucisson();

TestSlicing::TestSlicing(int deviceId) {
  this->deviceId=deviceId;

  TEST_ADD(TestSlicing::testPI);
}

void TestSlicing::testPI() {
  TEST_ASSERT(useSaucisson() == true);
}
