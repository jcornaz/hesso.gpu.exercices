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

  const int KERNEL_SIZE = kernelWidth * kernelHeight;
  const int IMAGE_SIZE = imageWidth * imageHeight;
  const int HALF_KERNEL_SIZE = KERNEL_SIZE / 2;
  const int HALF_KERNEL_WIDTH = kernelWidth / 2;

  uchar4 ptrResult[IMAGE_SIZE];

  #pragma omp parallel
  {
    const int TID = OmpTools::getTid();


    int s = TID;
    int i, j;
    float sum;
    while (s < IMAGE_SIZE) {
      IndiceTools::toIJ(s, imageWidth, &i, &j);

      if (i - HALF_KERNEL_WIDTH >= 0 && i + HALF_KERNEL_WIDTH < imageHeight && j - HALF_KERNEL_WIDTH >= 0 && j + HALF_KERNEL_WIDTH < imageWidth) {
        sum = 0.0;

        for (int v = 1 ; v <= HALF_KERNEL_WIDTH ; v++) {
          for (int u = 1 ; u <= HALF_KERNEL_WIDTH ; u++) {
            sum += ptrPixels[s + v * imageWidth + u].x * ptrKernel[HALF_KERNEL_SIZE + v * kernelWidth + u];
            sum += ptrPixels[s - v * imageWidth + u].x * ptrKernel[HALF_KERNEL_SIZE - v * kernelWidth + u];
            sum += ptrPixels[s + v * imageWidth - u].x * ptrKernel[HALF_KERNEL_SIZE + v * kernelWidth - u];
            sum += ptrPixels[s - v * imageWidth - u].x * ptrKernel[HALF_KERNEL_SIZE - v * kernelWidth - u];
          }

          sum += ptrPixels[s - v * imageWidth].x * ptrKernel[HALF_KERNEL_SIZE - v * kernelWidth];
          sum += ptrPixels[s + v * imageWidth].x * ptrKernel[HALF_KERNEL_SIZE + v * kernelWidth];
          sum += ptrPixels[s + v].x * ptrKernel[HALF_KERNEL_SIZE + v];
          sum += ptrPixels[s - v].x * ptrKernel[HALF_KERNEL_SIZE - v];
        }

        sum += ptrPixels[s].x * ptrKernel[HALF_KERNEL_SIZE];

        ptrResult[s].x = (int) sum;
        ptrResult[s].y = (int) sum;
        ptrResult[s].z = (int) sum;
      } else {
        ptrResult[s].x = 0;
        ptrResult[s].y = 0;
        ptrResult[s].z = 0;
      }

      ptrResult[s].w = 255;
      s += NB_THREADS;
    }
  }

  #pragma omp prallel for
  for (int i = 0 ; i < IMAGE_SIZE ; i++) {
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
