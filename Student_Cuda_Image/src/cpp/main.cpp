#include <iostream>
#include <stdlib.h>
#include <assert.h>

#include "cudaTools.h"
#include "Device.h"
#include "GLUTImageViewers.h"
#include "Option.h"

using std::cout;
using std::endl;

extern int mainGL(Option& option);
extern int mainFreeGL(Option& option);

int main(int argc, char** argv);


static int use(Option& option);
static int start(Option& option);
static void initCuda(Option& option);

int main(int argc, char** argv) {
  // Server Cuda1: in [0,5]
  // Server Cuda2: in [0,2]
  int DEVICE_ID = 0;
  bool IS_GL = false;

  Option option(IS_GL, DEVICE_ID,argc,argv);

  use(option);
}

int use(Option& option) {
  if (Device::isCuda()) {
      initCuda(option);
      int isOk = start(option);

      HANDLE_ERROR(cudaDeviceReset()); //cudaDeviceReset causes the driver to clean up all state. While not mandatory in normal operation, it is good practice.

      return isOk;
    } else {
      return EXIT_FAILURE;
    }
}

void initCuda(Option& option) {
  int deviceId = option.getDeviceId();

  // Check deviceId area
  int nbDevice = Device::getDeviceCount();
  assert(deviceId >= 0 && deviceId < nbDevice);

  // Choose current device  (state of host-thread)
  HANDLE_ERROR(cudaSetDevice(deviceId));

  // It can be usefull to preload driver, by example to practice benchmarking! (sometimes slow under linux)
  Device::loadCudaDriver(deviceId);
  // Device::loadCudaDriverAll();// Force driver to be load for all GPU

  // Enable p2p
  Device::p2pEnableALL();
}

int start(Option& option) {

	Device::printAll();
	Device::printAllSimple();
	Device::printCurrent();
	Device::print(option.getDeviceId());

if (option.isGL()) {
    return mainGL(option); // Bloquant, Tant qu'une fenetre est ouverte
  } else {
    return mainFreeGL(option);
  }
}
