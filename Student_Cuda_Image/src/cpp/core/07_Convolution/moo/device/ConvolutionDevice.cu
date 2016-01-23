#include "cudaType.h"
#include "Indice2D.h"
#include "IndiceTools.h"

__global__ void convolution(uchar4* ptrDevPixels, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight);
__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight);


__global__ void convolution(uchar4* ptrDevPixels, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight) {
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();
  const int N = imageWidth * imageHeight;

  const int DELTA_RIGHT = kernelWidth / 2;
  const int DELTA_LEFT = kernelWidth - DELTA_RIGHT;
  const int DELTA_DOWN = kernelHeight / 2;
  const int DELTA_UP = kernelHeight - DELTA_DOWN;

  int s = TID;
  int i, j;
  float sumX, sumY, sumZ;
  while (s < N) {
    IndiceTools::toIJ(s, imageWidth, &i, &j);

    ptrDevPixels[s].x = 255;
    ptrDevPixels[s].y = 0;
    ptrDevPixels[s].z = 0;

    if (i - DELTA_LEFT >= 0 && i + DELTA_RIGHT < imageWidth && j - DELTA_UP >= 0 && j + DELTA_DOWN < imageHeight) {
      sumX = 0.0;
      sumY = 0.0;
      sumZ = 0.0;

      int si, sk;
      for (int ik = 0 ; ik < kernelHeight ; ik++) {
        for (int jk = 0 ; jk < kernelWidth ; jk++) {
          si = IndiceTools::toS(imageWidth, i - DELTA_LEFT + ik, j - DELTA_UP + j);
          sk = IndiceTools::toS(kernelWidth, ik, jk);
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
