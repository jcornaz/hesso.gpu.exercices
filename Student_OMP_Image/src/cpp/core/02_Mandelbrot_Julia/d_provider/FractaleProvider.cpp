#include "FractaleProvider.h"
#include "FractaleMOO.h"
#include "Fractale.h"
#include "DomaineMath.h"
#include "Mandelbrot.h"
#include "ImageFonctionel.h"

ImageFonctionel* FractaleProvider::createGL() {
  AnimableFonctionel_I* ptrAnimable = FractaleProvider::createMOO();
  return new ImageFonctionel(ptrAnimable);
}

AnimableFonctionel_I* FractaleProvider::createMOO() {
  DomaineMath* domain = new DomaineMath(-2.1, -1.3, 0.8, 1.3);
  Mandelbrot* algo = new Mandelbrot();

  return new FractaleMOO(960, 960, domain, algo, 0, 100);
}
