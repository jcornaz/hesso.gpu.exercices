#include "Device.h"
#include "TestSlicing.h"

extern bool useSlicing();

TestSlicing::TestSlicing(int deviceId) {
  this->deviceId=deviceId;

  TEST_ADD(TestSlicing::testPI);
}

void TestSlicing::testPI() {
  TEST_ASSERT(useSlicing());
}
