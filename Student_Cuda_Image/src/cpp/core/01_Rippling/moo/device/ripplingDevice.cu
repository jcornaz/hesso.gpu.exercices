#include <iostream>

#include "cudaTools.h"
#include "Indice1D.h"
#include "Indice2D.h"
#include "Device.h"
#include "IndiceTools.h"
#include "RipplingMath.h"

__global__ void ripplingOneToOne(uchar4* ptrDevPixels, int w, int h, float t);
__global__ void ripplingOneDimension(uchar4* ptrDevPixels, int w, int h, float t);
__global__ void ripplingTwoDimensions(uchar4* ptrDevPixels, int w, int h, float t);

__global__ void ripplingOneToOne(uchar4* ptrDevPixels, int w, int h, float t) {
  RipplingMath ripplingMath(w, h);

  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int s = j + gridDim.x * blockDim.x * (threadIdx.y + blockIdx.y * blockDim.y);

  ripplingMath.colorIJ(&ptrDevPixels[s], i, j, t);
}

__global__ void ripplingOneDimension(uchar4* ptrDevPixels, int w, int h, float t) {
  RipplingMath ripplingMath(w, h);

  const int NB_THREADS = Indice1D::nbThread();
  const int TID = Indice1D::tid();
  const int n = w * h;
  int s = TID;
  while( s < n ) {
    int i, j;
    IndiceTools::toIJ(s, w, &i, &j);
    ripplingMath.colorIJ(&ptrDevPixels[s], i, j, t);
    s += NB_THREADS;
  }
}

__global__ void ripplingTwoDimensions(uchar4* ptrDevPixels, int w, int h, float t) {
  RipplingMath ripplingMath(w, h);

  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();
  const int n = w * h;
  int s = TID;
  while( s < n ) {
    int i, j;
    IndiceTools::toIJ(s, w, &i, &j);
    ripplingMath.colorIJ(&ptrDevPixels[s], i, j, t);
    s += NB_THREADS;
  }
}
