#include "ReductionTools.h"
#include "cudaTools.h"
#include "CustomMathTools.h"
#include "curand_kernel.h"

__global__ void computePIWithMonteCarlo(int* ptrDevResult, curandState* ptrDevTabGenerators, int nbGen);
__device__ int monteCarloIntraThreadReduction(curandState* ptrDevTabGenerators, int nbGen);
__device__ void generateRandomPoint(curandState* ptrDevGenerator, float xMin, float xMax, float yMin, float yMax, float* x, float* y);

__global__ void computePIWithMonteCarlo(int* ptrDevResult, curandState* ptrDevTabGenerators, int nbGen) {
  const int NB_THREADS_LOCAL = blockDim.x;
  const int TID_LOCAL = threadIdx.x;

  __shared__ int ptrDevArraySM[1024];

  ptrDevArraySM[TID_LOCAL] = monteCarloIntraThreadReduction(ptrDevTabGenerators, nbGen);
  __syncthreads();
  ReductionTools::template intraBlockReduction<int>(ptrDevArraySM, NB_THREADS_LOCAL);
  ReductionTools::template interBlocReduction<int>(ptrDevArraySM, NB_THREADS_LOCAL, ptrDevResult);
}

__device__ int monteCarloIntraThreadReduction(curandState* ptrDevTabGenerators, int nbGen) {
  const int NB_THREADS = gridDim.x * blockDim.x;
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;

  int threadSum = 0;
  float x, y;
  int s = TID;
  while (s < nbGen) {

    generateRandomPoint(&ptrDevTabGenerators[TID], 0, 1, 0, 4, &x, &y);

    if (y <= CustomMathTools::fpi(x)) {
      threadSum++;
    }

    s += NB_THREADS;
  }

  return threadSum;
}

__device__ void generateRandomPoint(curandState* ptrDevGenerator, float xMin, float xMax, float yMin, float yMax, float* x, float* y) {
  *x = curand_uniform(ptrDevGenerator) * (xMax - xMin) + xMin;
  *y = curand_uniform(ptrDevGenerator) * (yMax - yMin) + yMin;
}
