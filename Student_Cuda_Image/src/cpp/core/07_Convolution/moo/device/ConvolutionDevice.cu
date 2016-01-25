#include "cudaType.h"
#include "Indice1D.h"
#include "IndiceTools.h"

#define KERNEL_WIDTH 9
#define KERNEL_SIZE 81

__constant__ float KERNEL[KERNEL_SIZE];

__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight);
__global__ void convolution(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight);
__global__ void computeMinMax(uchar4* ptrDevPixels, int size, int* ptrDevMin, int* ptrDevMax);
__global__ void transform(uchar4* ptrDevPixels, int size, int* ptrDevBlack, int* ptrDevWhite);

float* getPtrDevKernel() {
  float* ptrDevKernel;
  HANDLE_ERROR(cudaGetSymbolAddress((void**) &ptrDevKernel, KERNEL));
  return ptrDevKernel;
}

__global__ void convolution(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();
  const int IMAGE_SIZE = imageWidth * imageHeight;
  const int HALF_KERNEL_SIZE = KERNEL_SIZE / 2;
  const int HALF_KERNEL_WIDTH = KERNEL_WIDTH / 2;

  int s = TID;
  int i, j;
  float sum;
  while (s < IMAGE_SIZE) {
    IndiceTools::toIJ(s, imageWidth, &i, &j);

    if (i - HALF_KERNEL_WIDTH >= 0 && i + HALF_KERNEL_WIDTH < imageHeight && j - HALF_KERNEL_WIDTH >= 0 && j + HALF_KERNEL_WIDTH < imageWidth) {
      sum = 0.0;

      for (int v = 1 ; v <= HALF_KERNEL_WIDTH ; v++) {
        for (int u = 1 ; u <= HALF_KERNEL_WIDTH ; u++) {
          sum += ptrDevPixels[s + v * imageWidth + u].x * KERNEL[HALF_KERNEL_SIZE + v * KERNEL_WIDTH + u];
          sum += ptrDevPixels[s - v * imageWidth + u].x * KERNEL[HALF_KERNEL_SIZE - v * KERNEL_WIDTH + u];
          sum += ptrDevPixels[s + v * imageWidth - u].x * KERNEL[HALF_KERNEL_SIZE + v * KERNEL_WIDTH - u];
          sum += ptrDevPixels[s - v * imageWidth - u].x * KERNEL[HALF_KERNEL_SIZE - v * KERNEL_WIDTH - u];
        }

        sum += ptrDevPixels[s - v * imageWidth].x * KERNEL[HALF_KERNEL_SIZE - v * KERNEL_WIDTH];
        sum += ptrDevPixels[s + v * imageWidth].x * KERNEL[HALF_KERNEL_SIZE + v * KERNEL_WIDTH];
        sum += ptrDevPixels[s + v].x * KERNEL[HALF_KERNEL_SIZE + v];
        sum += ptrDevPixels[s - v].x * KERNEL[HALF_KERNEL_SIZE - v];
      }

      sum += ptrDevPixels[s].x * KERNEL[HALF_KERNEL_SIZE];

      ptrDevResult[s].x = (int) sum;
      ptrDevResult[s].y = (int) sum;
      ptrDevResult[s].z = (int) sum;
    } else {
      ptrDevResult[s].x = 0;
      ptrDevResult[s].y = 0;
      ptrDevResult[s].z = 0;
    }

    ptrDevResult[s].w = 255;
    s += NB_THREADS;
  }
}

__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();
  const int N = imageWidth * imageHeight;

  int s = TID;
  while (s < N) {

    char grayLevel = (ptrDevPixels[s].x + ptrDevPixels[s].y + ptrDevPixels[s].z) / 3;

    ptrDevPixels[s].x = grayLevel;
    ptrDevPixels[s].y = grayLevel;
    ptrDevPixels[s].z = grayLevel;

    s += NB_THREADS;
  }
}

__global__ void computeMinMax(uchar4* ptrDevPixels, int size, int* ptrDevMin, int* ptrDevMax) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();

  // TODO
}

__global__ void transform(uchar4* ptrDevPixels, int size, int* ptrDevBlack, int* ptrDevWhite) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();

  // TODO
}
