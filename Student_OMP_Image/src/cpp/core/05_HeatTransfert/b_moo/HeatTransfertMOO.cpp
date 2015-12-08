#include <iostream>
#include <omp.h>

#include "HeatTransfertMOO.h"
#include "OmpTools.h"
#include "IndiceTools.h"

HeatTransfertMOO::HeatTransfertMOO(unsigned int w, unsigned int h, float* ptrImageInit, float* ptrImageHeater, float propSpeed) {

  // Inputs
  this->w = w;
  this->h = h;
  this->wh = w*h;

  // Images
  this->ptrImageInit = ptrImageInit;
  this->ptrImageHeater = ptrImageHeater;
  this->ptrImageA = (float*) malloc(sizeof(float) * wh);
  this->ptrImageB = (float*) malloc(sizeof(float) * wh);

  // Initialization
  this->crush(this->ptrImageHeater, this->ptrImageInit);
  this->diffuse(this->ptrImageInit, this->ptrImageA);
  this->crush(this->ptrImageHeater, this->ptrImageA);

  // Tools
  this->parallelPatern = OMP_MIXTE;
  this->iteration = 0;
  this->propSpeed = propSpeed;
}

HeatTransfertMOO::~HeatTransfertMOO(void) {
  free(this->ptrImageA);
  free(this->ptrImageB);
}

/**
 * Override
 */
void HeatTransfertMOO::process(uchar4* ptrPixels,int w,int h) {
  if (this->iteration % 2 == 0) {
    this->diffuse(this->ptrImageA, this->ptrImageB);
    this->crush(this->ptrImageHeater, this->ptrImageB);
    this->toScreen(this->ptrImageB, ptrPixels);
  } else {
    this->diffuse(this->ptrImageB, this->ptrImageA);
    this->crush(this->ptrImageHeater, this->ptrImageA);
    this->toScreen(this->ptrImageA, ptrPixels);
  }
}

/**
 * Override
 */
void HeatTransfertMOO::animationStep() {
  this->iteration++;
}

/**
 * Override
 */
float HeatTransfertMOO::getAnimationPara() {
  return this->iteration;
}

/**
 * Override
 */
int HeatTransfertMOO::getW() {
  return this->w;
}

/**
 * Override
 */
int HeatTransfertMOO::getH() {
  return this->h;
}

/**
 * Override
 */
string HeatTransfertMOO::getTitle() {
  return "Heat transfert";
}

void HeatTransfertMOO::setParallelPatern(ParallelPatern parallelPatern) {
  this->parallelPatern = parallelPatern;
}

void HeatTransfertMOO::diffuse(float* ptrImageInput, float* ptrImageOutput) {
  #pragma omp parallel
  {
    const unsigned int NB_THREADS = OmpTools::setAndGetNaturalGranularity();
    const unsigned int TID = OmpTools::getTid();
    unsigned int s = TID;

    while ( s < this->wh ) {
      int i, j;
			IndiceTools::toIJ(s, this->w, &i, &j);
      if (i > 0 && i < (this->h - 1) && j > 0 && j < (this->w - 1)) {

        float neighborsHeat[4];
        neighborsHeat[0] = ptrImageInput[IndiceTools::toS(this->w, i - 1, j)];
        neighborsHeat[1] = ptrImageInput[IndiceTools::toS(this->w, i + 1, j)];
        neighborsHeat[2] = ptrImageInput[IndiceTools::toS(this->w, i, j - 1)];
        neighborsHeat[3] = ptrImageInput[IndiceTools::toS(this->w, i, j + 1)];

        ptrImageOutput[s] = this->math.computeHeat(ptrImageInput[s], neighborsHeat, 4, this->propSpeed);
      } else {
        ptrImageOutput[s] = ptrImageInput[s];
      }

      s += NB_THREADS;
    }
  }
}

void HeatTransfertMOO::crush(float* ptrImageHeater, float* ptrImage) {
  #pragma omp parallel
  {
    const unsigned int NB_THREADS = OmpTools::setAndGetNaturalGranularity();
    const unsigned int TID = OmpTools::getTid();
    unsigned int s = TID;

    while ( s < this->wh ) {
      if (ptrImageHeater[s] > 0.0) {
        ptrImage[s] = ptrImageHeater[s];
      }
      s += NB_THREADS;
    }
  }
}

void HeatTransfertMOO::toScreen(float* ptrImage, uchar4* ptrPixels) {
  // TODO
}
