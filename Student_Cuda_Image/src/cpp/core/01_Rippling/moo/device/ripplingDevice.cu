#include <iostream>

#include "Indice2D.h"
#include "cudaTools.h"
#include "Device.h"
#include "IndiceTools.h"
#include "RipplingMath.h"

__global__ void rippling(uchar4* ptrDevPixels, int w, int h, float t);

__global__ void rippling(uchar4* ptrDevPixels, int w, int h, float t) {
  RipplingMath ripplingMath = RipplingMath(w, h);

  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int s = j + gridDim.x * (threadIdx.y + blockIdx.y * blockDim.y);

  ripplingMath.colorIJ(&ptrDevPixels[s], i, j, t);
}
