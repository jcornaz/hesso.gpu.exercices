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
  this->ptrDevImagesOutputs = (uchar4**) malloc(sizeof(uchar4*) * this->nbDevices);
  this->ptrDevMins = (int**) malloc(sizeof(int*) * this->nbDevices);
  this->ptrDevMaxs = (int**) malloc(sizeof(int*) * this->nbDevices);

  int imageWidth = this->videoCapter->getW();
  int imageHeight = this->videoCapter->getH();

  #pragma omp parallel for
  for(int deviceID = 0 ; deviceID < this->nbDevices ; deviceID++ )
  {

    HANDLE_ERROR(cudaSetDevice(deviceID));
    Device::assertDim(dg, db);

    // Compute the height of the image part to process
    int dividedImageHeight = imageHeight / this->nbDevices;
    if (deviceID == this->nbDevices - 1) {
      dividedImageHeight += imageHeight % this->nbDevices;
    }

    // Compute heightSupplement, index of the first pixel and the size of the image array
    // Theses all depend on the kernel size and on the deviceID
    int heightSupplement = (KERNEL_WIDTH * ((deviceID == 0 || deviceID == this->nbDevices - 1)? 1 : 2));
    int index = (deviceID * (imageHeight / this->nbDevices) * imageWidth) - (deviceID == 0 ? 0 : imageWidth * KERNEL_WIDTH);
    int size = (dividedImageHeight + heightSupplement)* imageWidth;

    // Copy the kernel in constant memory
    HANDLE_ERROR(cudaMemcpy(getPtrDevKernel(), ptrKernel, sizeof(float) * KERNEL_SIZE, cudaMemcpyHostToDevice));

    // Allocate global memories
    HANDLE_ERROR(cudaMalloc(&this->ptrDevMins[deviceID], sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevMaxs[deviceID], sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImages[deviceID], sizeof(uchar4) * size));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImagesOutputs[deviceID], sizeof(uchar4) * size));
  }
}

ConvolutionMOO::~ConvolutionMOO() {
  this->videoCapter->stop();

  #pragma omp parallel for
  for (int deviceID = 0 ; deviceID < this->nbDevices ; deviceID++ ) {
    HANDLE_ERROR(cudaFree(this->ptrDevImages[deviceID]));
    HANDLE_ERROR(cudaFree(this->ptrDevImagesOutputs[deviceID]));
    HANDLE_ERROR(cudaFree(this->ptrDevMins[deviceID]));
    HANDLE_ERROR(cudaFree(this->ptrDevMaxs[deviceID]));
  }

  free(this->videoCapter);
  free(this->ptrDevImages);
  free(this->ptrDevImagesOutputs);
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

  int minimums[this->nbDevices];
  int maximums[this->nbDevices];
  int baseMin = 255;
  int baseMax = 0;

  #pragma omp parallel for
  for(int deviceID = 0 ; deviceID < this->nbDevices ; deviceID++ )
  {
    HANDLE_ERROR(cudaSetDevice(deviceID));

    // Compute the height of the image part to process
    int dividedImageHeight = h / this->nbDevices;
    if (deviceID == this->nbDevices - 1) {
      dividedImageHeight += h % this->nbDevices;
    }

    // Compute heightSupplement, index of the first pixel and the size of the image array
    // Theses all depend on the kernel size and on the deviceID
    int heightSupplement = (KERNEL_WIDTH * ((deviceID == 0 || deviceID == this->nbDevices - 1)? 1 : 2));
    int index = (deviceID * (h / this->nbDevices) * w) - (deviceID == 0 ? 0 : w * KERNEL_WIDTH);
    int size = (dividedImageHeight + heightSupplement)* w;

    // Copy the image part to process to the device
    HANDLE_ERROR(cudaMemcpy(this->ptrDevImages[deviceID], &ptrImage[index], sizeof(uchar4) * size, cudaMemcpyHostToDevice));

    // Convert in black and white
    convertInBlackAndWhite<<<dg,db>>>(this->ptrDevImages[deviceID], size);

    // Apply the convolution algorithm
    convolution<<<dg,db>>>(this->ptrDevImages[deviceID], this->ptrDevImagesOutputs[deviceID], w, dividedImageHeight + heightSupplement);

    // Prepare the minimum and maximum
    HANDLE_ERROR(cudaMemcpy(this->ptrDevMins[deviceID], &baseMin, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->ptrDevMaxs[deviceID], &baseMax, sizeof(int), cudaMemcpyHostToDevice));

    // Copmute the min and max pixel values
    computeMinMax<<<dg,db>>>(this->ptrDevImagesOutputs[deviceID], size, this->ptrDevMins[deviceID], this->ptrDevMaxs[deviceID]);

    // Retrieve the min and max values
    HANDLE_ERROR(cudaMemcpy(&minimums[deviceID], this->ptrDevMins[deviceID], sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&maximums[deviceID], this->ptrDevMaxs[deviceID], sizeof(int), cudaMemcpyDeviceToHost));

    // Copy the pixels to the displayable device
    HANDLE_ERROR(cudaMemcpy(&ptrDevPixels[deviceID * (h / this->nbDevices) * w], this->ptrDevImagesOutputs[deviceID], sizeof(uchar4) * dividedImageHeight * w, cudaMemcpyDeviceToDevice));
  }

  // inter-GPU min-max reduction
  int min = 255;
  int max = 0;
  for(int deviceID = 0 ; deviceID < this->nbDevices ; deviceID++) {
    if (minimums[deviceID] < min) { min = minimums[deviceID]; }
    if (maximums[deviceID] > max) { max = maximums[deviceID]; }
  }

  // Apply an affine transform on the pixels
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
