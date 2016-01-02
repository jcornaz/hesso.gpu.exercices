#include "ReductionTools.h"
#include "cudaTools.h"

__global__ void computePIWithSlicing(float* ptrDevResult, int nbSlices);
__device__ float intraThreadReduction(int nbSlices);
__device__ float fpi(float x);

/**
 * Calcul la valeur de pi par découpage de l'intégrale bornée de [0,1] de f(x) = 4 / (1 + x*x) <br />
 * Préconditions :
 * <ul>
 *  <li>Le nombre de thread par bloc doit être une puissance de deux</li>
 *  <li>La grille et les blocs doivent être à une dimension (en x)
 * </ul>
 * @param ptrDevResult Pointeur d'un emplacement mémoire (sur le device) où déposer le résultat
 * @param nbSlices Nombre de découpage de l'intégrale. Plus ce nombre est grand, plus la précision sera grande et le calcul lent.
 */
__global__ void computePIWithSlicing(float* ptrDevResult, int nbSlices) {
  const int NB_THREADS_LOCAL = blockDim.x;
  const int TID_LOCAL = threadIdx.x;

  __shared__ float ptrDevArraySM[1024];

  ptrDevArraySM[TID_LOCAL] = intraThreadReduction(nbSlices);
  __syncthreads();
  ReductionTools::template intraBlockReduction<float>(ptrDevArraySM, NB_THREADS_LOCAL);
  ReductionTools::template interBlocReduction<float>(ptrDevArraySM, NB_THREADS_LOCAL, ptrDevResult);
}

__device__ float intraThreadReduction(int nbSlices) {
  const int NB_THREADS = gridDim.x * blockDim.x;
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;

  float dx = 1. / nbSlices;

  float threadSum = 0.0;
  int s = TID;
  while (s < nbSlices) {
    threadSum += fpi(s * dx);
    s += NB_THREADS;
  }

  return threadSum / nbSlices;
}

__device__ float fpi(float x) {
  return 4 / (1 + x * x);
}
