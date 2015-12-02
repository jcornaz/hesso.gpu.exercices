#include "Indice2D.h"
#include "cudaTools.h"

__device__ void ecrasement(float* arraySM, int size) {
  // Intra-block

  const int NB_THREAD_LOCAL = blockDim.x * blockDim.y;
  const int TID_LOCAL = blockId.x + blockId.y * blockDim.x;

  int n = size;
  int half = size / 2;
  while (half >= 1) {

    int s = TID_LOCAL;
    while (s < half) {
      arraySM[s] += arraySM[s + half];
      s += NB_THREAD_LOCAL;
    }

    __syncthreads();

    n = hafl;
    half = n / 2;
  }
}

__device__ void interbloc(float* arraySM, int size, float* resultGM) {
  // inter-block

  const int NB_THREAD_LOCAL = blockDim.x * blockDim.y;
  const int TID_LOCAL = blockId.x + blockId.y * blockDim.x;

  if (TID_LOCAL == 0) {
    atomicAdd(resultGM, arraySM[0]);
  }
}
