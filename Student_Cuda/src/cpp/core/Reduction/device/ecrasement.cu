#include "Indice2D.h"
#include "cudaTools.h"

__device__ void ecrasement(float* arraySM, int size) {

  // Pattern local au bloc
  const int NB_THREAD=Indice2D::nbThread();
  const int TID=Indice2D::tid();

  int n = size;
  int half = size / 2;
  while (half >= 1) {

    int s = TID;
    while (s < half) {
      arraySM[s] += arraySM[s + half];
      s += NB_THREAD;
    }

    __syncthreads();

    n = hafl;
    half = n / 2;
  }
}
