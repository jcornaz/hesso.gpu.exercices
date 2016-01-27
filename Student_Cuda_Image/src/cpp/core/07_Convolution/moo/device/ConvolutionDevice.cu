#include "cudaType.h"
#include "Indice1D.h"
#include "IndiceTools.h"
#include "ConvolutionConstants.h"

texture<uchar4, 2> textureRef;

__constant__ float KERNEL[KERNEL_SIZE];

__host__ void initTexture();
__host__ void bindTexture(uchar4* ptrDevPixels, int width, int heitht);
__host__ void unbindTexture();
__host__ float* getPtrDevKernel();

__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int size);
__global__ void convolution(uchar4* ptrDevResult, int imageWidth, int imageHeight);
__global__ void computeMinMax(uchar4* ptrDevPixels, int size, int* ptrDevMin, int* ptrDevMax);
__global__ void transform(uchar4* ptrDevPixels, int size, int black, int white);

__device__ void intraThreadMinMaxReduction(int* minimumsArraySM, int* maximumsArraySM, uchar4* ptrDevPixels, int imageSize);
__device__ void intraBlockMinMaxReduction(int* minimumsArraySM, int* maximumsArraySM, int arraySize);
__device__ void interBlockMinMaxReduction(int* minimumsArraySM, int* maximumsArraySM, int* minimumResult, int* maximumResult);

__host__ void initTexture() {
  // textureRef.addressMode[0] = cudaAddressModeMirror;
  // textureRef.addressMode[1] = cudaAddressModeMirror;
}

__host__ void bindTexture(uchar4* ptrDevPixels, int width, int height) {
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
  cudaBindTexture2D(NULL, textureRef, ptrDevPixels, channelDesc, width, height, sizeof(uchar4) * width);
}

__host__ void unbindTexture() {
  cudaUnbindTexture(textureRef);
}

__host__ float* getPtrDevKernel() {
  float* ptrDevKernel;
  HANDLE_ERROR(cudaGetSymbolAddress((void**) &ptrDevKernel, KERNEL));
  return ptrDevKernel;
}

__global__ void convolution(uchar4* ptrDevResult, int imageWidth, int imageHeight) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();
  const int IMAGE_SIZE = imageWidth * imageHeight;
  const int HALF_KERNEL_SIZE = KERNEL_SIZE / 2;
  const int HALF_KERNEL_WIDTH = KERNEL_WIDTH / 2;

  int s = TID;
  int i, j;
  float sum;
  if (s < IMAGE_SIZE) {
    IndiceTools::toIJ(s, imageWidth, &i, &j);
    sum = 0.0;

    for (int v = 1 ; v <= HALF_KERNEL_WIDTH ; v++) {
      for (int u = 1 ; u <= HALF_KERNEL_WIDTH ; u++) {
        sum += tex2D(textureRef, j + u, i + v).x * KERNEL[HALF_KERNEL_SIZE + v * KERNEL_WIDTH + u];
        sum += tex2D(textureRef, j + u, i - v).x * KERNEL[HALF_KERNEL_SIZE - v * KERNEL_WIDTH + u];
        sum += tex2D(textureRef, j - u, i + v).x * KERNEL[HALF_KERNEL_SIZE + v * KERNEL_WIDTH - u];
        sum += tex2D(textureRef, j - u, i - v).x * KERNEL[HALF_KERNEL_SIZE - v * KERNEL_WIDTH - u];
      }

      sum += tex2D(textureRef, j, i - v).x * KERNEL[HALF_KERNEL_SIZE - v * KERNEL_WIDTH];
      sum += tex2D(textureRef, j, i + v).x * KERNEL[HALF_KERNEL_SIZE + v * KERNEL_WIDTH];
      sum += tex2D(textureRef, j - v, i).x * KERNEL[HALF_KERNEL_SIZE - v];
      sum += tex2D(textureRef, j + v, i).x * KERNEL[HALF_KERNEL_SIZE + v];
    }

    sum += tex2D(textureRef, j, i).x * KERNEL[HALF_KERNEL_SIZE];

    ptrDevResult[s].x = (int) sum;
    ptrDevResult[s].y = (int) sum;
    ptrDevResult[s].z = (int) sum;

    ptrDevResult[s].w = 255;
    s += NB_THREADS;
  }
}

__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int size) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();

  int s = TID;
  if (s < size) {
    char grayLevel = (ptrDevPixels[s].x + ptrDevPixels[s].y + ptrDevPixels[s].z) / 3;

    ptrDevPixels[s].x = grayLevel;
    ptrDevPixels[s].y = grayLevel;
    ptrDevPixels[s].z = grayLevel;

    s += NB_THREADS;
  }
}

__global__ void transform(uchar4* ptrDevPixels, int size, int black, int white) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();

  int delta = abs(white - black);

  int s = TID;
  int newValue;
  if (s < size) {
    newValue = (ptrDevPixels[s].x - black) * delta + black;
    ptrDevPixels[s].x = newValue;
    ptrDevPixels[s].y = newValue;
    ptrDevPixels[s].z = newValue;
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

__device__ void intraThreadMinMaxReduction(int* minimumsArraySM, int* maximumsArraySM, uchar4* ptrDevPixels, int imageSize) {
  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();

  int s = TID;
  int min = 255;
  int max = 0;
  int value;

  if (s < imageSize) {
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
