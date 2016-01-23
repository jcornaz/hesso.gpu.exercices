#include <iostream>
#include <stdlib.h>

#include "Settings.h"

#include "Animateur.h"
#include "AnimateurFonctionel.h"

#include "ConvolutionProvider.h"

using std::cout;
using std::endl;
using std::string;

int mainMOO(Settings& settings);

static void animeAndDestroy(Animable_I* ptrAnimable, int nbIteration);
static void animeAndDestroy(AnimableFonctionel_I* ptrAnimable, int nbIteration);

int mainMOO(Settings& settings) {
  cout << "\n[FreeGL] mode" << endl;

  const int NB_ITERATION = 1000;

  animeAndDestroy(ConvolutionProvider::createMOO(), NB_ITERATION);

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
