#ifndef REDUCTION_H_
#define REDUCTION_H_

#include "cudaTools.h"

template<typename T>
__device__ void intraBlockReduction(T* arraySM, int size) {
  const int NB_THREAD_LOCAL = blockDim.x * blockDim.y;
  const int TID_LOCAL = blockIdx.x + blockIdx.y * blockDim.x;

  int n = size;
  int half = size / 2;
  while (half >= 1) {

    int s = TID_LOCAL;
    while (s < half) {
      arraySM[s] += arraySM[s + half];
      s += NB_THREAD_LOCAL;
    }

    __syncthreads();

    n = half;
    half = n / 2;
  }
}

template<typename T>
__device__ void interBlocReduction(T* arraySM, int size, T* resultGM) {
  const int NB_THREAD_LOCAL = blockDim.x * blockDim.y;
  const int TID_LOCAL = blockIdx.x + blockIdx.y * blockDim.x;

  if (TID_LOCAL == 0) {
    atomicAdd(resultGM, arraySM[0]);
  }
}

#endif
