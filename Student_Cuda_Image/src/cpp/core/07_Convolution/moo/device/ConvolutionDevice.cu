#include "cudaType.h"
#include "Indice2D.h"
#include "IndiceTools.h"

__global__ void convolution(uchar4* ptrDevPixels, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight);
__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight);


__global__ void convolution(uchar4* ptrDevPixels, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight) {
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();
  const int N = imageWidth * imageHeight;

  int s = TID;
  int i, j;
  float sumX, sumY, sumZ;
  while (s < N) {
    IndiceTools::toIJ(s, imageWidth, &i, &j);

    if (i >= kernelHeight / 2 && i < imageHeight - kernelHeight / 2 && j >= kernelWidth / 2 && j < imageWidth - kernelWidth / 2) {
      sumX = 0.0;
      sumY = 0.0;
      sumZ = 0.0;

      int si, sk;
      for (int ii = 0 ; ii < kernelHeight ; ii++) {
        for (int jj = 0 ; jj < kernelWidth ; jj++) {
          si = IndiceTools::toS(imageWidth, i - ii, j - jj);
          sk = IndiceTools::toS(kernelWidth, ii, jj);
          sumX += ptrDevPixels[si].x * ptrDevKernel[sk];
          sumY += ptrDevPixels[si].y * ptrDevKernel[sk];
          sumZ += ptrDevPixels[si].z * ptrDevKernel[sk];
        }
      }

      ptrDevPixels[s].x = (int) (sumX / (kernelWidth * kernelHeight));
      ptrDevPixels[s].y = (int) (sumY / (kernelWidth * kernelHeight));
      ptrDevPixels[s].z = (int) (sumZ / (kernelWidth * kernelHeight));
      ptrDevPixels[s].w = 255;
    }

    s += NB_THREADS;
  }
}

__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight) {
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();
  const int N = imageWidth * imageHeight;

  int s = TID;
  while (s < N) {
    
    char grayLevel = (ptrDevPixels[s].x + ptrDevPixels[s].y + ptrDevPixels[s].z) / 3;

    ptrDevPixels[s].x = grayLevel;
    ptrDevPixels[s].y = grayLevel;
    ptrDevPixels[s].z = grayLevel;

    s += NB_THREADS;
  }
}
