#include <omp.h>

#include "HeatTransfertMOO.h"
#include "OmpTools.h"
#include "IndiceTools.h"


HeatTransfertMOO::HeatTransfertMOO(unsigned int w, unsigned int h, float* imageInit, float* heaters) {
  // Inputs
  this->w = w;
  this->h = h;

  // Tools
  this->parallelPatern = OMP_MIXTE;
}

HeatTransfertMOO::~HeatTransfertMOO(void) {
}

/**
 * Override
 */
void HeatTransfertMOO::process(uchar4* ptrTabPixels,int w,int h) {
  switch (parallelPatern) {
    case OMP_ENTRELACEMENT: // Plus lent sur CPU
      entrelacementOMP(ptrTabPixels, w, h);
      break;

    case OMP_FORAUTO: // Plus rapide sur CPU
      forAutoOMP(ptrTabPixels, w, h);
      break;

    case OMP_MIXTE: // Pour tester que les deux implementations fonctionnent
      // Note : Des saccades peuvent apparaitre à cause de la grande difference de fps entre la version entrelacer et auto
      static bool isEntrelacement = true;
      if (isEntrelacement) {
        entrelacementOMP(ptrTabPixels, w, h);
      } else {
        forAutoOMP(ptrTabPixels, w, h);
      }
      isEntrelacement = !isEntrelacement; // Pour swithcer a chaque iteration
      break;
  }
}

/**
 * Override
 */
void HeatTransfertMOO::animationStep() {
  // TODO
}

/**
 * Override
 */
float HeatTransfertMOO::getAnimationPara() {
  // TODO
  return 1;
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

/**
 * Code entrainement Cuda
 */
void HeatTransfertMOO::entrelacementOMP(uchar4* ptrTabPixels, int w, int h) {

  const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();

  #pragma omp parallel
  {
    const int TID = OmpTools::getTid();
    const int n = w * h;
    int s = TID;
    while ( s < n ) {
      int i, j;
      IndiceTools::toIJ(s, this->w, &i, &j);
      // TODO
      s++;
    }
  }
}

/**
 * Code naturel et direct OMP
 */
void HeatTransfertMOO::forAutoOMP(uchar4* ptrTabPixels, int w, int h) {
  const int n = w * h;

  #pragma omp parallel for
  for (int s = 0; s < n; s ++) {
    int i, j;
    IndiceTools::toIJ(s, this->w, &i, &j);
    // TODO
  }
}
