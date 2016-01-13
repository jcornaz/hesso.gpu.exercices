#include "ConvolutionMOO.h"
#include "Device.h"

extern __global__ void convolution(uchar4* ptrDevPixels, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight);

ConvolutionMOO::ConvolutionMOO(int w, int h, ConvolutionKernel& kernel_) : kernel(kernel_) {
  this->w = w;
  this->h = h;
  this->t = 0;

  this->dg = dim3(64, 64, 1);
  this->db = dim3(32, 32, 1);

  Device::assertDim(dg, db);

  HANDLE_ERROR(cudaMalloc(&this->ptrDevKernel, sizeof(float) * kernel.getSize()));
  HANDLE_ERROR(cudaMemcpy(this->ptrDevKernel, kernel.getWeights(), sizeof(float) * kernel.getSize(), cudaMemcpyHostToDevice));
}

ConvolutionMOO::~ConvolutionMOO() {
  HANDLE_ERROR(cudaFree(this->ptrDevKernel));
}

/**
* Call periodicly by the api
*/
void ConvolutionMOO::process(uchar4* ptrDevPixels, int w, int h) {
  convolution<<<dg,db>>>(ptrDevPixels, this->w, this->h, this->ptrDevKernel, this->kernel.getWidth(), this->kernel.getHeight());
}

/**
* Call periodicly by the api
*/
void ConvolutionMOO::animationStep() {
  this->t++;
}

float ConvolutionMOO::getAnimationPara() {
  return this->t;
}

string ConvolutionMOO::getTitle() {
  return "Convolution";
}

int ConvolutionMOO::getW() {
  return this->w;
}

int ConvolutionMOO::getH() {
  return this->h;
}
