#include "ConvolutionMOO.h"

ConvolutionMOO::ConvolutionMOO(int w, int h, ConvolutionKernel& kernel) {
  this->w = w;
  this->h = h;
  this->t = 0;
}

ConvolutionMOO::~ConvolutionMOO() {
  // Nothing to do
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
