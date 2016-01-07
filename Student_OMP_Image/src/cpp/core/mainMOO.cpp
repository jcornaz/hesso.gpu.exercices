#include <iostream>
#include <stdlib.h>

#include "Settings.h"

#include "Animateur.h"
#include "AnimateurFonctionel.h"

#include "RipplingProvider.h"
#include "MandelbrotProvider.h"
#include "JuliaProvider.h"
#include "NewtonProvider.h"
#include "RayTracingProvider.h"
#include "HeatTransfertProvider.h"

using std::cout;
using std::endl;
using std::string;

int mainMOO(Settings& settings);

static void animeAndDestroy(Animable_I* ptrAnimable, int nbIteration);
static void animeAndDestroy(AnimableFonctionel_I* ptrAnimable, int nbIteration);

int mainMOO(Settings& settings) {
  cout << "\n[FreeGL] mode" << endl;

  const int NB_ITERATION = 1000;

  animeAndDestroy(RipplingProvider::createMOO(), NB_ITERATION);
  animeAndDestroy(MandelbrotProvider::createMOO(), NB_ITERATION);
  animeAndDestroy(JuliaProvider::createMOO(), NB_ITERATION);
  animeAndDestroy(NewtonProvider::createMOO(), NB_ITERATION);
  animeAndDestroy(RayTracingProvider::createMOO(), NB_ITERATION);
  animeAndDestroy(HeatTransfertProvider::createMOO(), NB_ITERATION);

  cout << "\n[FreeGL] end" << endl;

  return EXIT_SUCCESS;
}

void animeAndDestroy(Animable_I* ptrAnimable, int nbIteration) {
  Animateur animateur(ptrAnimable, nbIteration);
  animateur.run();

  delete ptrAnimable;
}

void animeAndDestroy(AnimableFonctionel_I* ptrAnimable, int nbIteration) {
  AnimateurFonctionel animateur(ptrAnimable, nbIteration);
  animateur.run();

  delete ptrAnimable;
}
