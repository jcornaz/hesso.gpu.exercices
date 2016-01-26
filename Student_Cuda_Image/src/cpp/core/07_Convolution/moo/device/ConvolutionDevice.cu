#include "cudaType.h"
#include "Indice1D.h"
#include "IndiceTools.h"
#include "ConvolutionConstants.h"

__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int size);
__global__ void convolution(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight);
__global__ void computeMinMax(uchar4* ptrDevPixels, int size, int* ptrDevMin, int* ptrDevMax);
__global__ void transform(uchar4* ptrDevPixels, int size, int* ptrDevBlack, int* ptrDevWhite);

__device__ void intraThreadMinMaxReduction(int* minimumsArraySM, int* maximumsArraySM, uchar4* ptrDevPixels, int imageSize);
__device__ void intraBlockMinMaxReduction(int* minimumsArraySM, int* maximumsArraySM, int arraySize);
__device__ void interBlockMinMaxReduction(int* minimumsArraySM, int* maximumsArraySM, int* minimumResult, int* maximumResult);

__global__ void convolution(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();
  const int SIZE_IMAGE = imageWidth * imageHeight;
  const int SIZE_KERNEL = kernelWidth * kernelHeight;

  const int DELTA_RIGHT = kernelWidth / 2;
  const int DELTA_LEFT = kernelWidth - DELTA_RIGHT;
  const int DELTA_DOWN = kernelHeight / 2;
  const int DELTA_UP = kernelHeight - DELTA_DOWN;

  int s = TID;
  int i, j, si, sk, ik, jk;
  float sumX, sumY, sumZ;
  while (s < SIZE_IMAGE) {
    IndiceTools::toIJ(s, imageWidth, &i, &j);

    if (i - DELTA_UP >= 0 && i + DELTA_DOWN < imageHeight && j - DELTA_LEFT >= 0 && j + DELTA_RIGHT < imageWidth) {
      sumX = 0.0;
      sumY = 0.0;
      sumZ = 0.0;

      sk = 0;
      while (sk < SIZE_KERNEL) {
        IndiceTools::toIJ(sk, kernelWidth, &ik, &jk);
        si = IndiceTools::toS(imageWidth, i - DELTA_UP + ik, j - DELTA_LEFT + jk);
        sumX += ptrDevPixels[si].x * ptrDevKernel[sk];
        sumY += ptrDevPixels[si].y * ptrDevKernel[sk];
        sumZ += ptrDevPixels[si].z * ptrDevKernel[sk];
        sk++;
      }

      ptrDevResult[s].x = (int) sumX;
      ptrDevResult[s].y = (int) sumY;
      ptrDevResult[s].z = (int) sumZ;
    } else {
      ptrDevResult[s].x = 0;
      ptrDevResult[s].y = 0;
      ptrDevResult[s].z = 0;
    }

    ptrDevResult[s].w = 255;
    s += NB_THREADS;
  }
}

__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int size) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();

  int s = TID;
  while (s < size) {

    char grayLevel = (ptrDevPixels[s].x + ptrDevPixels[s].y + ptrDevPixels[s].z) / 3;

    ptrDevPixels[s].x = grayLevel;
    ptrDevPixels[s].y = grayLevel;
    ptrDevPixels[s].z = grayLevel;

    s += NB_THREADS;
  }
}

__global__ void computeMinMax(uchar4* ptrDevPixels, int imageSize, int* ptrDevMin, int* ptrDevMax) {
  __shared__ int ptrDevMinimumsSM[NB_THREADS_BY_BLOCK];
  __shared__ int ptrDevMaximumsSM[NB_THREADS_BY_BLOCK];

  intraThreadMinMaxReduction(ptrDevMinimumsSM, ptrDevMaximumsSM, ptrDevPixels, imageSize);
  __syncthreads();
  intraBlockMinMaxReduction(ptrDevMinimumsSM, ptrDevMaximumsSM, NB_THREADS_BY_BLOCK);
  interBlockMinMaxReduction(ptrDevMinimumsSM, ptrDevMaximumsSM, ptrDevMin, ptrDevMax);
}

__global__ void transform(uchar4* ptrDevPixels, int size, int* ptrDevBlack, int* ptrDevWhite) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();

  int black = *ptrDevBlack;
  int white = *ptrDevWhite;
  int delta = abs(white - black);

  int s = TID;
  int newValue;
  while (s < size) {
    newValue = (ptrDevPixels[s].x - black) * delta + black;
    ptrDevPixels[s].x = newValue;
    ptrDevPixels[s].y = newValue;
    ptrDevPixels[s].z = newValue;
    s += NB_THREADS;
  }
}

__device__ void intraThreadMinMaxReduction(int* minimumsArraySM, int* maximumsArraySM, uchar4* ptrDevPixels, int imageSize) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();

  int s = TID;
  int min = 255;
  int max = 0;
  int value;
  while(s < imageSize) {
    value = ptrDevPixels[s].x;
    if (value < min) { min = value; }
    if (value > max) { max = value; }
    s += NB_THREADS;
  }

  minimumsArraySM[threadIdx.x] = min;
  maximumsArraySM[threadIdx.x] = max;
}

__device__ void intraBlockMinMaxReduction(int* minimumsArraySM, int* maximumsArraySM, int arraySize) {
  const int NB_THREADS_LOCAL = blockDim.x;
  const int TID_LOCAL = threadIdx.x;

  int n = arraySize;
  int half = arraySize / 2;
  while (half >= 1) {

    int s = TID_LOCAL;
    while (s < half) {

      if (minimumsArraySM[s + half] < minimumsArraySM[s]) {
        minimumsArraySM[s] = minimumsArraySM[s + half];
      }

      if (maximumsArraySM[s + half] > maximumsArraySM[s]) {
        maximumsArraySM[s] = maximumsArraySM[s + half];
      }

      s += NB_THREADS_LOCAL;
    }

    __syncthreads();

    n = half;
    half = n / 2;
  }
}

__device__ void interBlockMinMaxReduction(int* minimumsArraySM, int* maximumsArraySM, int* minimumResult, int* maximumResult) {
  if (threadIdx.x == 0) {
    atomicMin(minimumResult, minimumsArraySM[0]);
    atomicMax(maximumResult, maximumsArraySM[0]);
  }
}
