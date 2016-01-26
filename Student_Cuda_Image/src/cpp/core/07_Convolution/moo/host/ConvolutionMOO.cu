#include "ConvolutionMOO.h"
#include "Device.h"
#include "OpencvTools.h"
#include "cudaType.h"
#include "ConvolutionConstants.h"

extern __global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int size);
extern __global__ void convolution(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight);
extern __global__ void computeMinMax(uchar4* ptrDevPixels, int size, int* ptrDevMin, int* ptrDevMax);
extern __global__ void transform(uchar4* ptrDevPixels, int size, int black, int white);
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

  this->nbDevices = Device::getDeviceCount();

  this->ptrDevImages = (uchar4**) malloc(sizeof(uchar4*) * this->nbDevices);
  this->ptrDevMins = (int**) malloc(sizeof(int*) * this->nbDevices);
  this->ptrDevMaxs = (int**) malloc(sizeof(int*) * this->nbDevices);

  int imageWidth = this->videoCapter->getW();
  int imageHeight = this->videoCapter->getH();

  #pragma omp parallel for
  for(int deviceID = 0 ; deviceID < this->nbDevices ; deviceID++ )
  {
    Device::assertDim(dg, db);

    HANDLE_ERROR(cudaSetDevice(deviceID));

    HANDLE_ERROR(cudaMalloc(&this->ptrDevMins[deviceID], sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevMaxs[deviceID], sizeof(int)));

    int dividedImageHeight = imageHeight / this->nbDevices;
    if (deviceID == this->nbDevices - 1) {
      dividedImageHeight += imageHeight % this->nbDevices;
    }

    uchar4* ptrDev;
    HANDLE_ERROR(cudaMalloc(&ptrDev, sizeof(uchar4) * imageWidth * dividedImageHeight));
    this->ptrDevImages[deviceID] = ptrDev;
  }

  HANDLE_ERROR(cudaSetDevice(0));

  HANDLE_ERROR(cudaMalloc(&this->ptrDevImage, sizeof(uchar4) * imageWidth * imageHeight));
  HANDLE_ERROR(cudaMemcpy(getPtrDevKernel(), ptrKernel, sizeof(float) * KERNEL_SIZE, cudaMemcpyHostToDevice));
}

ConvolutionMOO::~ConvolutionMOO() {
  this->videoCapter->stop();

  #pragma omp parallel for
  for (int deviceID = 0 ; deviceID < this->nbDevices ; deviceID++ ) {
    HANDLE_ERROR(cudaFree(this->ptrDevImages[deviceID]));
    HANDLE_ERROR(cudaFree(this->ptrDevMins[deviceID]));
    HANDLE_ERROR(cudaFree(this->ptrDevMaxs[deviceID]));
  }
  HANDLE_ERROR(cudaFree(this->ptrDevImage));

  free(this->videoCapter);
  free(this->ptrDevImages);
  free(this->ptrDevMins);
  free(this->ptrDevMaxs);
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

  HANDLE_ERROR(cudaMemcpy(this->ptrDevImage, ptrImage, sizeof(uchar4) * imageSize, cudaMemcpyHostToDevice));

  convertInBlackAndWhite<<<dg,db>>>(this->ptrDevImage, imageSize);
  convolution<<<dg,db>>>(this->ptrDevImage, ptrDevPixels, w, h);

  int minimums[this->nbDevices];
  int maximums[this->nbDevices];
  int baseMin = 255;
  int baseMax = 0;

  #pragma omp parallel for
  for(int deviceID = 0 ; deviceID < this->nbDevices ; deviceID++ )
  {
    HANDLE_ERROR(cudaSetDevice(deviceID));

    int dividedImageHeight = h / this->nbDevices;
    if (deviceID == this->nbDevices - 1) {
      dividedImageHeight += h % this->nbDevices;
    }

    HANDLE_ERROR(cudaMemcpy(this->ptrDevImages[deviceID], &ptrDevPixels[deviceID * (h / this->nbDevices) * w], sizeof(uchar4) * dividedImageHeight * w, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->ptrDevMins[deviceID], &baseMin, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->ptrDevMaxs[deviceID], &baseMax, sizeof(int), cudaMemcpyHostToDevice));
    computeMinMax<<<dg,db>>>(this->ptrDevImages[deviceID], dividedImageHeight, this->ptrDevMins[deviceID], this->ptrDevMaxs[deviceID]);
    HANDLE_ERROR(cudaMemcpy(&minimums[deviceID], this->ptrDevMins[deviceID], sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&maximums[deviceID], this->ptrDevMaxs[deviceID], sizeof(int), cudaMemcpyDeviceToHost));
  }

  // Réduction inter-GPU
  int min = 255;
  int max = 0;
  for(int deviceID = 0 ; deviceID < this->nbDevices ; deviceID++) {
    if (minimums[deviceID] < min) { min = minimums[deviceID]; }
    if (maximums[deviceID] > max) { max = maximums[deviceID]; }
  }

  HANDLE_ERROR(cudaSetDevice(0));
  transform<<<dg,db>>>(ptrDevPixels, imageSize, max, min);
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
