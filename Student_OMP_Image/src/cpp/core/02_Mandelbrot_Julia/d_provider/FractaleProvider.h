#ifndef FRACTALE_PROVIDER_H_
#define FRACTALE_PROVIDER_H_

#include "ImageFonctionel.h"
#include "AnimableFonctionel_I.h"

class FractaleProvider {

  public:
  	static ImageFonctionel* createGL();
  	static AnimableFonctionel_I* createMOO();
};

#endif
