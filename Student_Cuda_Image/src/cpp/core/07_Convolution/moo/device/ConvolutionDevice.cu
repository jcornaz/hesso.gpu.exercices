#include "cudaType.h"
#include "Indice2D.h"
#include "IndiceTools.h"

__global__ void convolution(uchar4* ptrDevPixels, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight);
__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight);


__global__ void convolution(uchar4* ptrDevPixels, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight) {
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();
  const int SIZE_IMAGE = imageWidth * imageHeight;
  const int SIZE_KERNEL = kernelWidth * kernelHeight;

  const int DELTA_RIGHT = kernelWidth / 2;
  const int DELTA_LEFT = kernelWidth - DELTA_RIGHT;
  const int DELTA_DOWN = kernelHeight / 2;
  const int DELTA_UP = kernelHeight - DELTA_DOWN;

  int s = TID;
  int i, j, si, sk, ik, jk;
  float sumX, sumY, sumZ;
  while (s < SIZE_IMAGE) {
    IndiceTools::toIJ(s, imageWidth, &i, &j);

    if (i - DELTA_UP >= 0 && i + DELTA_DOWN < imageHeight && j - DELTA_LEFT >= 0 && j + DELTA_RIGHT < imageWidth) {
      sumX = 0.0;
      sumY = 0.0;
      sumZ = 0.0;

      sk = 0;
      while (sk < SIZE_KERNEL) {
        IndiceTools::toIJ(sk, kernelWidth, &ik, &jk);
        si = IndiceTools::toS(imageWidth, i - DELTA_UP + ik, j - DELTA_LEFT + jk);
        sumX += ptrDevPixels[si].x * ptrDevKernel[sk];
        sumY += ptrDevPixels[si].y * ptrDevKernel[sk];
        sumZ += ptrDevPixels[si].z * ptrDevKernel[sk];
        sk++;
      }

      ptrDevPixels[s].x = (int) sumX;
      ptrDevPixels[s].y = (int) sumY;
      ptrDevPixels[s].z = (int) sumZ;
    }

    ptrDevPixels[s].w = 255;
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
