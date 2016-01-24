#include "cudaType.h"
#include "Indice2D.h"
#include "IndiceTools.h"

__global__ void convolution(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight);
__global__ void transform(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight, int kernelWidth);
__global__ void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight);

__global__ void convolution(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight) {
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
  float sum;
  while (s < SIZE_IMAGE) {
    IndiceTools::toIJ(s, imageWidth, &i, &j);

    if (i - DELTA_UP >= 0 && i + DELTA_DOWN < imageHeight && j - DELTA_LEFT >= 0 && j + DELTA_RIGHT < imageWidth) {
      sum = 0.0;

      sk = 0;
      while (sk < SIZE_KERNEL) {
        IndiceTools::toIJ(sk, kernelWidth, &ik, &jk);
        si = IndiceTools::toS(imageWidth, i - DELTA_UP + ik, j - DELTA_LEFT + jk);
        sum += ptrDevPixels[si].x * ptrDevKernel[sk];
        sk++;
      }

      ptrDevResult[s].x = (int) sum;
      ptrDevResult[s].y = (int) sum;
      ptrDevResult[s].z = (int) sum;
    } else {
      ptrDevResult[s].x = 0;
      ptrDevResult[s].y = 0;
      ptrDevResult[s].z = 0;
    }

    ptrDevResult[s].w = 255;
    s += NB_THREADS;
  }
}

__global__ void transform(uchar4* ptrDevPixels, uchar4* ptrDevResult, int imageWidth, int imageHeight, int kernelWidth) {
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();
  const int SIZE_IMAGE = imageWidth * imageHeight;
  const int SIZE_KERNEL = kernelWidth * kernelWidth;

  const int DELTA_RIGHT = kernelWidth / 2;
  const int DELTA_LEFT = kernelWidth - DELTA_RIGHT;
  const int DELTA_DOWN = DELTA_RIGHT;
  const int DELTA_UP = DELTA_LEFT;


  int s = TID;
  int i, j, si, sk, ik, jk;
  int xmin, ymin, zmin;
  int xmax, ymax, zmax;
  while (s < SIZE_IMAGE) {
    IndiceTools::toIJ(s, imageWidth, &i, &j);

    if (i - DELTA_UP >= 0 && i + DELTA_DOWN < imageHeight && j - DELTA_LEFT >= 0 && j + DELTA_RIGHT < imageWidth) {
      xmin = 256;
      ymin = 256;
      zmin = 256;
      xmax = -1;
      ymax = -1;
      zmax = -1;

      sk = 0;
      while (sk < SIZE_KERNEL) {
        IndiceTools::toIJ(sk, kernelWidth, &ik, &jk);
        si = IndiceTools::toS(imageWidth, i - DELTA_UP + ik, j - DELTA_LEFT + jk);

        if (ptrDevPixels[si].x < xmin) { xmin = ptrDevPixels[si].x; }
        if (ptrDevPixels[si].x > xmax) { xmax = ptrDevPixels[si].x; }
        if (ptrDevPixels[si].y < ymin) { ymin = ptrDevPixels[si].y; }
        if (ptrDevPixels[si].y > ymax) { ymax = ptrDevPixels[si].y; }
        if (ptrDevPixels[si].z < zmin) { zmin = ptrDevPixels[si].z; }
        if (ptrDevPixels[si].z > zmax) { zmax = ptrDevPixels[si].z; }

        sk++;
      }

      float coeff = (xmax - xmin) / 255.0;

      ptrDevResult[s].x = (int) (255 - ((ptrDevPixels[s].x - xmin) * coeff + xmin));
      ptrDevResult[s].y = (int) (255 - ((ptrDevPixels[s].y - xmin) * coeff + xmin));
      ptrDevResult[s].z = (int) (255 - ((ptrDevPixels[s].z - xmin) * coeff + xmin));
    } else {
      ptrDevResult[s].x = 0;
      ptrDevResult[s].y = 0;
      ptrDevResult[s].z = 0;
    }

    ptrDevResult[s].w = 255;
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
