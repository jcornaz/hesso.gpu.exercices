#include "ConvolutionMOO.h"
#include "OpencvTools.h"
#include "IndiceTools.h"

ConvolutionMOO::ConvolutionMOO(string videoPath, int kernelWidth, int kernelHeight, float* ptrKernel) {

  this->t = 0;
  this->kernelWidth = kernelWidth;
  this->kernelHeight = kernelHeight;

  this->videoCapter = new CVCaptureVideo("/media/Data/Video/autoroute.mp4");

  size_t size = sizeof(float) * kernelWidth * kernelHeight;
  this->ptrKernel = (float*) malloc(size);

  int n = kernelWidth * kernelHeight;
  for (int i = 0 ; i < n ; i++) {
    this->ptrKernel[i] = ptrKernel[i];
  }

  this->videoCapter->start();
}

ConvolutionMOO::~ConvolutionMOO() {
  this->videoCapter->stop();
  free(this->videoCapter);
  free(this->ptrKernel);
}

void ConvolutionMOO::setParallelPatern(ParallelPatern pattern) {
  // Nothing to do, because this class is a sequential version for speedups
}

/**
* Call periodicly by the api
*/
void ConvolutionMOO::process(uchar4* ptrPixels, int w, int h) {
  Mat matRGBA(h, w, CV_8UC1);
  Mat matBGR = this->videoCapter->provideBGR();
  OpencvTools::switchRB(matRGBA, matBGR);
  uchar4* ptrImage = OpencvTools::castToUchar4(matRGBA);

  int n = w * h;
  for (int i = 0 ; i < n ; i++) {
    ptrPixels[i] = ptrImage[i];
  }

  int black, white;

  this->convertInBlackAndWhite(ptrPixels, w, h);
  this->convolution(ptrPixels, w, h, this->ptrKernel, this->kernelWidth, this->kernelHeight);
  this->computeMinMax(ptrPixels, n, &white, &black);
  this->transform(ptrPixels, n, black, white);
}

/**
* Call periodicly by the api
*/
void ConvolutionMOO::animationStep() {
  this->t++;
}

float ConvolutionMOO::getAnimationPara() {
  return this->t;
}

string ConvolutionMOO::getTitle() {
  return "Convolution";
}

int ConvolutionMOO::getW() {
  return this->videoCapter->getW();
}

int ConvolutionMOO::getH() {
  return this->videoCapter->getH();
}

void ConvolutionMOO::convolution(uchar4* ptrPixels, int imageWidth, int imageHeight, float* ptrKernel, int kernelWidth, int kernelHeight) {
  const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();

  const int SIZE_IMAGE = imageWidth * imageHeight;
  const int SIZE_KERNEL = kernelWidth * kernelHeight;

  const int DELTA_RIGHT = kernelWidth / 2;
  const int DELTA_LEFT = kernelWidth - DELTA_RIGHT;
  const int DELTA_DOWN = kernelHeight / 2;
  const int DELTA_UP = kernelHeight - DELTA_DOWN;

  uchar4 ptrResult[SIZE_IMAGE];

  #pragma omp parallel
  {
    const int TID = OmpTools::getTid();

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
          sumX += ptrPixels[si].x * ptrKernel[sk];
          sumY += ptrPixels[si].y * ptrKernel[sk];
          sumZ += ptrPixels[si].z * ptrKernel[sk];
          sk++;
        }

        ptrResult[s].x = (int) sumX;
        ptrResult[s].y = (int) sumY;
        ptrResult[s].z = (int) sumZ;
      }

      ptrPixels[s].w = 255;
      s += NB_THREADS;
    }
  }

  #pragma omp prallel for
  for (int i = 0 ; i < SIZE_IMAGE ; i++) {
    ptrPixels[i].x = ptrResult[i].x;
    ptrPixels[i].y = ptrResult[i].y;
    ptrPixels[i].z = ptrResult[i].z;
  }
}

void ConvolutionMOO::convertInBlackAndWhite(uchar4* ptrPixels, int imageWidth, int imageHeight) {
  const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();

  #pragma omp parallel
  {
    const int N = imageWidth * imageHeight;
    const int TID = OmpTools::getTid();

    int s = TID;
    while (s < N) {
      char grayLevel = (ptrPixels[s].x + ptrPixels[s].y + ptrPixels[s].z) / 3;

      ptrPixels[s].x = grayLevel;
      ptrPixels[s].y = grayLevel;
      ptrPixels[s].z = grayLevel;

      s += NB_THREADS;
    }
  }
}

void ConvolutionMOO::computeMinMax(uchar4* ptrPixels, int imageSize, int* ptrMin, int* ptrMax) {
  const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();

  int minimumsArray[NB_THREADS];
  int maximumsArray[NB_THREADS];

  #pragma omp parallel
  {
    const int TID = OmpTools::getTid();

    int s = TID;
    int min = 255;
    int max = 0;
    int value;
    while(s < imageSize) {
      value = ptrPixels[s].x;
      if (value < min) { min = value; }
      if (value > max) { max = value; }
      s += NB_THREADS;
    }

    minimumsArray[TID] = min;
    maximumsArray[TID] = max;
  }

  int min = 255;
  int max = 0;
  for (int i = 0 ; i < NB_THREADS ; i++) {
    if (minimumsArray[i] < min) { min = minimumsArray[i]; }
    if (maximumsArray[i] > max) { max = maximumsArray[i]; }
  }

  *ptrMin = min;
  *ptrMax = max;
}

void ConvolutionMOO::transform(uchar4* ptrPixels, int size, int black, int white) {
  const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();

  #pragma omp parallel
  {
    const int TID = OmpTools::getTid();

    int delta = abs(white - black);

    int s = TID;
    int newValue;
    while (s < size) {
      newValue = (ptrPixels[s].x - black) * delta + black;
      ptrPixels[s].x = newValue;
      ptrPixels[s].y = newValue;
      ptrPixels[s].z = newValue;
      s += NB_THREADS;
    }
  }
}
