#ifndef REDUCTION_H_
#define REDUCTION_H_

#include "cudaTools.h"

class ReductionTools {

  public:
    template<typename T>
    __device__ static void intraBlockReduction(T* arraySM, int size) {
      const int NB_THREADS_LOCAL = blockDim.x * blockDim.y * blockDim.z;
      const int TID_LOCAL = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

      int n = size;
      int half = size / 2;
      while (half >= 1) {

        int s = TID_LOCAL;
        while (s < half) {
          arraySM[s] += arraySM[s + half];
          s += NB_THREADS_LOCAL;
        }

        __syncthreads();

        n = half;
        half = n / 2;
      }
    }

    template<typename T>
    __device__ static void interBlocReduction(T* arraySM, int size, T* resultGM) {
      const int TID_LOCAL = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

      if (TID_LOCAL == 0) {
        atomicAdd(resultGM, arraySM[0]);
      }
    }
};

#endif
