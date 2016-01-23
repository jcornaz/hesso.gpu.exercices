#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "Option.h"

#include "Animateur.h"
#include "AnimateurFonctionel.h"

#include "ConvolutionProvider.h"

using std::cout;
using std::endl;
using std::string;

int mainFreeGL(Option& option);

static void animeAndDelete(Animable_I* ptrAnimable, int nbIteration);
static void animeAndDelete(AnimableFonctionel_I* ptrAnimable, int nbIteration);

int mainFreeGL(Option& option) {

  cout << "\n[FreeGL] mode" << endl;

  const int NB_ITERATION = 1000;

  animeAndDelete(ConvolutionProvider::createMOO(), NB_ITERATION);

  cout << "\n[FreeGL] end" << endl;

  return EXIT_SUCCESS;
}

void animeAndDelete(Animable_I* ptrAnimable, int nbIteration) {
  Animateur animateur(ptrAnimable, nbIteration);
  animateur.run();

  delete ptrAnimable;
}

void animeAndDelete(AnimableFonctionel_I* ptrAnimable, int nbIteration) {
  AnimateurFonctionel animateur(ptrAnimable, nbIteration);
  animateur.run();

  delete ptrAnimable;
}
