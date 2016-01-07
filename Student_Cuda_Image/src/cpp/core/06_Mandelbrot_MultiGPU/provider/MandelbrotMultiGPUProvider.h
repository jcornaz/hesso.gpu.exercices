#ifndef MANDELBROT_MULTI_GPU_PROVIDER_H_
#define MANDELBROT_MULTI_GPU_PROVIDER_H_

#include "ImageFonctionel.h"
#include "AnimableFonctionel_I.h"

class MandelbrotMultiGPUProvider {

  public:
  	static ImageFonctionel* createGL();
  	static AnimableFonctionel_I* createMOO();
};

#endif
