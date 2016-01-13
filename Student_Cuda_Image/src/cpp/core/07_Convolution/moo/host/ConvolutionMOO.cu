#include "ConvolutionMOO.h"

ConvolutionMOO::ConvolutionMOO(int w, int h, ConvolutionKernel& kernel_) : kernel(kernel_) {
  this->w = w;
  this->h = h;
  this->t = 0;

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
  // TODO
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
