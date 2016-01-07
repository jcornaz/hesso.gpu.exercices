#ifndef MANDELBROT_PROVIDER_H_
#define MANDELBROT_PROVIDER_H_

#include "ImageFonctionel.h"
#include "AnimableFonctionel_I.h"

class MandelbrotProvider {

  public:
  	static ImageFonctionel* createGL();
  	static AnimableFonctionel_I* createMOO();
};

#endif
