#include <iostream>
#include <stdlib.h>

#include "Settings.h"

#include "Animateur.h"
#include "AnimateurFonctionel.h"

#include "RayTracingProvider.h"

using std::cout;
using std::endl;
using std::string;

int mainMOO(Settings& settings);

static void animer(Animable_I* ptrAnimable, int nbIteration);
static void animer(AnimableFonctionel_I* ptrAnimable, int nbIteration);

int mainMOO(Settings& settings) {
  cout << "\n[FreeGL] mode" << endl;

  const int NB_ITERATION = 1000;

  Animable_I* ptrRayTracing = RayTracingProvider::createMOO();
  animer(ptrRayTracing, NB_ITERATION);
  delete ptrRayTracing;

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
