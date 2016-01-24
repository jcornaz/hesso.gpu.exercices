#include "ConvolutionMOO.h"
#include "Device.h"
#include "OpencvTools.h"
#include "cudaType.h"

extern __global__ void convolution(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight);
extern __global__ void transform(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight, int kernelWidth);
extern __global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight);

ConvolutionMOO::ConvolutionMOO(string videoPath, int kernelWidth, int kernelHeight, float* ptrKernel) {

  this->dg = dim3(64, 64, 1);
  this->db = dim3(32, 32, 1);
  Device::assertDim(dg, db);

  this->t = 0;
  this->kernelWidth = kernelWidth;
  this->kernelHeight = kernelHeight;
  this->videoCapter = new CVCaptureVideo("/media/Data/Video/autoroute.mp4");
  this->videoCapter->start();

  size_t kernelSize = sizeof(float) * kernelWidth * kernelHeight;
  HANDLE_ERROR(cudaMalloc(&this->ptrDevKernel, kernelSize));
  HANDLE_ERROR(cudaMalloc(&this->ptrDevImage, sizeof(uchar4) * this->videoCapter->getW() * this->videoCapter->getH()));
  HANDLE_ERROR(cudaMemcpy(this->ptrDevKernel, ptrKernel, kernelSize, cudaMemcpyHostToDevice));

  std::cout << this->videoCapter->getW() << ", " << this->videoCapter->getH() << std::endl;
}

ConvolutionMOO::~ConvolutionMOO() {
  this->videoCapter->stop();
  free(this->videoCapter);
  HANDLE_ERROR(cudaFree(this->ptrDevKernel));
  HANDLE_ERROR(cudaFree(this->ptrDevImage));
}

/**
* Call periodicly by the api
*/
void ConvolutionMOO::process(uchar4* ptrDevPixels, int w, int h) {
  Mat matRGBA(h, w, CV_8UC4);
  Mat matBGR = this->videoCapter->provideBGR();
  OpencvTools::switchRB(matRGBA, matBGR);
  uchar4* ptrImage = OpencvTools::castToUchar4(matRGBA);

  HANDLE_ERROR(cudaMemcpy(this->ptrDevImage, ptrImage, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
  
  convertInBlackAndWhite<<<dg,db>>>(this->ptrDevImage, w, h);
  convolution<<<dg,db>>>(this->ptrDevImage, ptrDevPixels, w, h, this->ptrDevKernel, this->kernelWidth, this->kernelHeight);

  HANDLE_ERROR(cudaMemcpy(this->ptrDevImage, ptrDevPixels, sizeof(uchar4) * w * h, cudaMemcpyDeviceToDevice));

  transform<<<dg,db>>>(this->ptrDevImage, ptrDevPixels, w, h, 3);
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
