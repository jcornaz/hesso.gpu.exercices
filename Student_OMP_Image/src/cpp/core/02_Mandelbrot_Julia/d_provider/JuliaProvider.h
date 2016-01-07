#ifndef JULIA_PROVIDER_H_
#define JULIA_PROVIDER_H_

#include "ImageFonctionel.h"
#include "AnimableFonctionel_I.h"

class JuliaProvider {

  public:
  	static ImageFonctionel* createGL();
  	static AnimableFonctionel_I* createMOO();
};

#endif
