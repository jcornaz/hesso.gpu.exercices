#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "Option.h"

#include "Animateur.h"
#include "AnimateurFonctionel.h"

#include "RipplingProvider.h"

using std::cout;
using std::endl;
using std::string;

int mainFreeGL(Option& option);

static void animer(Animable_I* ptrAnimable, int nbIteration);
static void animer(AnimableFonctionel_I* ptrAnimable, int nbIteration);

int mainFreeGL(Option& option) {

  cout << "\n[FreeGL] mode" << endl;

  const int NB_ITERATION = 1000;

  Animable_I* ptrAnimable = RipplingProvider::createMOO();
  animer(ptrAnimable, NB_ITERATION);
  delete ptrAnimable;

  cout << "\n[FreeGL] end" << endl;

  return EXIT_SUCCESS;
}

void animer(Animable_I* ptrAnimable, int nbIteration) {
  Animateur animateur(ptrAnimable, nbIteration);
  animateur.run();
}

void animer(AnimableFonctionel_I* ptrAnimable, int nbIteration) {
  AnimateurFonctionel animateur(ptrAnimable, nbIteration);
  animateur.run();
}
