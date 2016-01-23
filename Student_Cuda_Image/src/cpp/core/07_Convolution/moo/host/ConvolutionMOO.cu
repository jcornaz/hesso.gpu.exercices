#include "ConvolutionMOO.h"
#include "Device.h"
#include "OpencvTools.h"

extern __global__ void convolution(uchar4* ptrDevPixels, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight);
extern __global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight);

ConvolutionMOO::ConvolutionMOO(ConvolutionKernel& kernel_, string videoPath) :
    kernel(kernel_)
    {


  this->dg = dim3(64, 64, 1);
  this->db = dim3(32, 32, 1);

  this->videoCapter = new CVCaptureVideo("/media/Data/Video/autoroute.mp4");
  this->t = 0;

  Device::assertDim(dg, db);

  HANDLE_ERROR(cudaMalloc(&this->ptrDevKernel, sizeof(float) * kernel.getSize()));
  HANDLE_ERROR(cudaMemcpy(this->ptrDevKernel, kernel.getWeights(), sizeof(float) * kernel.getSize(), cudaMemcpyHostToDevice));

  this->videoCapter->start();
}

ConvolutionMOO::~ConvolutionMOO() {
  this->videoCapter->stop();
  free(this->videoCapter);
  HANDLE_ERROR(cudaFree(this->ptrDevKernel));
}

/**
* Call periodicly by the api
*/
void ConvolutionMOO::process(uchar4* ptrDevPixels, int w, int h) {
  Mat matRGBA(h, w, CV_8UC1);
  Mat matBGR = this->videoCapter->provideBGR();
  OpencvTools::switchRB(matRGBA, matBGR);
  uchar4* ptrImage = OpencvTools::castToUchar4(matRGBA);
  HANDLE_ERROR(cudaMemcpy(ptrDevPixels, ptrImage, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

  convertInBlackAndWhite<<<dg,db>>>(ptrDevPixels, w, h);
  convolution<<<dg,db>>>(ptrDevPixels, w, h, this->ptrDevKernel, this->kernel.getWidth(), this->kernel.getHeight());
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
  return this->videoCapter->getW();
}

int ConvolutionMOO::getH() {
  return this->videoCapter->getH();
}
