#include <iostream>
#include <omp.h>

#include "cuda_runtime.h"
#include "Device.h"
#include "HeatTransfert.h"
#include "IndiceTools.h"

__global__ void diffuse(float* ptrImageInput, float* ptrImageOutput, unsigned int w, unsigned int h, float propSpeed);
__global__ void crush(float* ptrImageHeater, float* ptrImage, unsigned int size);
__global__ void toScreen(float* ptrImage, uchar4* ptrPixels, unsigned int size);

HeatTransfert::HeatTransfert(unsigned int w, unsigned int h, float* ptrImageInit, float* ptrImageHeater, float propSpeed){

  // Inputs
  this->w = w;
  this->h = h;
  this->wh = w*h;

  // Tools
  this->iteration = 0;
  this->propSpeed = propSpeed;

  // Cuda grid dimensions
  this->dg = dim3(64, 64, 1);
  this->db = dim3(32, 32, 1);
  Device::assertDim(dg, db);

  size_t arraySize = sizeof(float) * wh;
  HANDLE_ERROR(cudaMalloc(&this->ptrDevImageHeater, arraySize));
  HANDLE_ERROR(cudaMalloc(&this->ptrDevImageInit, arraySize));
  HANDLE_ERROR(cudaMalloc(&this->ptrDevImageA, arraySize));
  HANDLE_ERROR(cudaMalloc(&this->ptrDevImageB, arraySize));
  HANDLE_ERROR(cudaMemcpy(this->ptrDevImageHeater, ptrImageHeater, arraySize, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->ptrDevImageInit, ptrImageInit, arraySize, cudaMemcpyHostToDevice));

  // Initialization
  crush<<<dg,db>>>(this->ptrDevImageHeater, this->ptrDevImageInit, this->wh);
  diffuse<<<dg,db>>>(this->ptrDevImageInit, this->ptrDevImageA, this->w, this->h, this->propSpeed);
  crush<<<dg,db>>>(this->ptrDevImageHeater, this->ptrDevImageA, this->wh);
}

HeatTransfert::~HeatTransfert() {
  HANDLE_ERROR(cudaFree(this->ptrDevImageHeater));
  HANDLE_ERROR(cudaFree(this->ptrDevImageInit));
  HANDLE_ERROR(cudaFree(this->ptrDevImageA));
  HANDLE_ERROR(cudaFree(this->ptrDevImageB));
}

/**
 * Override
 */
void HeatTransfert::process(uchar4* ptrDevPixels,int w,int h) {
  if (this->iteration % 2 == 0) {
    diffuse<<<dg,db>>>(this->ptrDevImageA, this->ptrDevImageB, this->w, this->h, this->propSpeed);
    crush<<<dg,db>>>(this->ptrDevImageHeater, this->ptrDevImageB, this->wh);
    toScreen<<<dg,db>>>(this->ptrDevImageB, ptrDevPixels, this->wh);
  } else {
    diffuse<<<dg,db>>>(this->ptrDevImageB, this->ptrDevImageA, this->w, this->h, this->propSpeed);
    crush<<<dg,db>>>(this->ptrDevImageHeater, this->ptrDevImageA, this->wh);
    toScreen<<<dg,db>>>(this->ptrDevImageA, ptrDevPixels, this->wh);
  }
}

/**
 * Override
 */
void HeatTransfert::animationStep() {
  this->iteration++;
}

/**
 * Override
 */
float HeatTransfert::getAnimationPara() {
  return this->iteration;
}

/**
 * Override
 */
int HeatTransfert::getW() {
  return this->w;
}

/**
 * Override
 */
int HeatTransfert::getH() {
  return this->h;
}

/**
 * Override
 */
string HeatTransfert::getTitle() {
  return "Heat transfert Cuda";
}
