#include "ConvolutionMOO.h"
#include "Device.h"
#include "OpencvTools.h"
#include "cudaType.h"

extern __global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight);
extern __global__ void convolution(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight);
extern __global__ void computeMinMax(uchar4* ptrDevPixels, int size, int* ptrDevMin, int* ptrDevMax);
extern __global__ void transform(uchar4* ptrDevPixels, int size, int* ptrDevBlack, int* ptrDevWhite);
extern float* getPtrDevKernel();

ConvolutionMOO::ConvolutionMOO(string videoPath, float* ptrKernel, int cudaGridDim, int cudaBlockDim) {

  this->dg = dim3(cudaGridDim, 1, 1);
  this->db = dim3(cudaBlockDim, 1, 1);
  Device::assertDim(dg, db);

  this->t = 0;
  this->videoCapter = new CVCaptureVideo("/media/Data/Video/autoroute.mp4");
  this->videoCapter->start();

  HANDLE_ERROR(cudaMalloc(&this->ptrDevMin, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&this->ptrDevMax, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&this->ptrDevImage, sizeof(uchar4) * this->videoCapter->getW() * this->videoCapter->getH()));
  HANDLE_ERROR(cudaMemcpy(getPtrDevKernel(), ptrKernel, sizeof(float) * 81, cudaMemcpyHostToDevice));
}

ConvolutionMOO::~ConvolutionMOO() {
  this->videoCapter->stop();
  free(this->videoCapter);
  HANDLE_ERROR(cudaFree(this->ptrDevImage));
  HANDLE_ERROR(cudaFree(this->ptrDevMin));
  HANDLE_ERROR(cudaFree(this->ptrDevMax));
}

/**
* Call periodicly by the api
*/
void ConvolutionMOO::process(uchar4* ptrDevPixels, int w, int h) {
  Mat matRGBA(h, w, CV_8UC4);
  Mat matBGR = this->videoCapter->provideBGR();
  OpencvTools::switchRB(matRGBA, matBGR);
  uchar4* ptrImage = OpencvTools::castToUchar4(matRGBA);
  const int IMAGE_SIZE = w * h;

  HANDLE_ERROR(cudaMemcpy(this->ptrDevImage, ptrImage, sizeof(uchar4) * IMAGE_SIZE, cudaMemcpyHostToDevice));

  convertInBlackAndWhite<<<dg,db>>>(this->ptrDevImage, w, h);
  convolution<<<dg,db>>>(this->ptrDevImage, ptrDevPixels, w, h);

  HANDLE_ERROR(cudaMemset(this->ptrDevMin, 0, sizeof(int)));
  HANDLE_ERROR(cudaMemset(this->ptrDevMax, 0, sizeof(int)));

  computeMinMax<<<dg,db>>>(ptrDevPixels, IMAGE_SIZE, this->ptrDevMin, this->ptrDevMax);
  transform<<<dg,db>>>(ptrDevPixels, IMAGE_SIZE, this->ptrDevMin, this->ptrDevMax);
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
