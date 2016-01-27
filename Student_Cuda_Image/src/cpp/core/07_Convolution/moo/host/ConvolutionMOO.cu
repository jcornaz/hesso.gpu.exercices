#include "ConvolutionMOO.h"
#include "Device.h"
#include "OpencvTools.h"
#include "cudaType.h"
#include "ConvolutionConstants.h"

extern __global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int size);
extern __global__ void convolution(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight);
extern __global__ void computeMinMax(uchar4* ptrDevPixels, int size, int* ptrDevMin, int* ptrDevMax);
extern __global__ void transform(uchar4* ptrDevPixels, int size, int* ptrDevBlack, int* ptrDevWhite);
extern float* getPtrDevKernel();

ConvolutionMOO::ConvolutionMOO(string videoPath, float* ptrKernel) {
  this->t = 0;
  this->videoCapter = new CVCaptureVideo("/media/Data/Video/autoroute.mp4");
  this->videoCapter->start();

  // Prepare Cuda GRID
  int size = this->videoCapter->getW() * this->videoCapter->getH();
  int gridDim = (size / NB_THREADS_BY_BLOCK) + ((size % NB_THREADS_BY_BLOCK == 0) ? 0 : 1);
  this->dg = dim3(gridDim, 1, 1);
  this->db = dim3(NB_THREADS_BY_BLOCK, 1, 1);
  Device::assertDim(dg, db);

  // Allocate memory in GPU
  HANDLE_ERROR(cudaMalloc(&this->ptrDevMin, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&this->ptrDevMax, sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&this->ptrDevImage, sizeof(uchar4) * size));
  HANDLE_ERROR(cudaMemcpy(getPtrDevKernel(), ptrKernel, sizeof(float) * KERNEL_SIZE, cudaMemcpyHostToDevice));
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
  int imageSize = w * h;

  int firstMin = 255;
  int firstMax = 0;

  HANDLE_ERROR(cudaMemcpy(this->ptrDevImage, ptrImage, sizeof(uchar4) * imageSize, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->ptrDevMin, &firstMin, sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(this->ptrDevMax, &firstMax, sizeof(int), cudaMemcpyHostToDevice));

  convertInBlackAndWhite<<<dg,db>>>(this->ptrDevImage, imageSize);
  convolution<<<dg,db>>>(this->ptrDevImage, ptrDevPixels, w, h);
  computeMinMax<<<dg,db>>>(ptrDevPixels, imageSize, this->ptrDevMin, this->ptrDevMax);

  transform<<<dg,db>>>(ptrDevPixels, imageSize, this->ptrDevMax, this->ptrDevMin);
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
