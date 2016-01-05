#ifndef NEWTON_PROVIDER_H_
#define NEWTON_PROVIDER_H_

#include "ImageFonctionel.h"
#include "AnimableFonctionel_I.h"

class NewtonProvider {

  public:
  	static ImageFonctionel* createGL();
  	static AnimableFonctionel_I* createMOO();
};

#endif
