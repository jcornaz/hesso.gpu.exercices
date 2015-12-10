#include <iostream>

#include "cudaTools.h"
#include "Device.h"

extern __global__ void computePI(float* ptrDevResult, int nbSlices);

bool isPIok();

bool isPIok() {

  dim3 dg = dim3(128, 1, 1);
  dim3 db = dim3(512, 1, 1);

  Device::assertDim(dg, db);

  float piValue;
  float* ptrDevResult;

  HANDLE_ERROR(cudaMalloc(&ptrDevResult, sizeof(float)));
  HANDLE_ERROR(cudaMemset(ptrDevResult, 0, sizeof(float)));

  computePI<<<dg,db>>>(ptrDevResult, 1000000);

  HANDLE_ERROR(cudaMemcpy(&piValue, ptrDevResult, sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(ptrDevResult));

  std::cout << "PI = " << piValue << std::endl;

  return abs(piValue - 3.141592653589793f) < 0.0000000001;
}
