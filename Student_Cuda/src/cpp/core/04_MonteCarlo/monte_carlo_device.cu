#include "ReductionTools.h"
#include "cudaTools.h"
#include "CustomMathTools.h"

__global__ void computePIWithMonteCarlo(float* ptrDevResult, int nbSlices);
__device__ float monteCarloIntraThreadReduction(int nbSlices);

__global__ void computePIWithMonteCarlo(float* ptrDevResult, int nbGenerations) {
  const int NB_THREADS_LOCAL = blockDim.x;
  const int TID_LOCAL = threadIdx.x;

  __shared__ float ptrDevArraySM[1024];

  ptrDevArraySM[TID_LOCAL] = monteCarloIntraThreadReduction(nbGenerations);
  __syncthreads();
  ReductionTools::template intraBlockReduction<float>(ptrDevArraySM, NB_THREADS_LOCAL);
  ReductionTools::template interBlocReduction<float>(ptrDevArraySM, NB_THREADS_LOCAL, ptrDevResult);
}

__device__ float monteCarloIntraThreadReduction(int nbSlices) {
  const int NB_THREADS = gridDim.x * blockDim.x;
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;

  float dx = 1. / nbSlices;

  float threadSum = 0.0;
  int s = TID;
  while (s < nbSlices) {
    threadSum += CustomMathTools::fpi(s * dx);
    s += NB_THREADS;
  }

  return threadSum / nbSlices;
}
