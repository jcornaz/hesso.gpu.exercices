#include "MandelbrotProvider.h"
#include "FractaleMOO.h"
#include "Mandelbrot.h"
#include "DomaineMath.h"

ImageFonctionel* MandelbrotProvider::createGL() {
  AnimableFonctionel_I* ptrAnimable = MandelbrotProvider::createMOO();
  return new ImageFonctionel(ptrAnimable);
}

AnimableFonctionel_I* MandelbrotProvider::createMOO() {
  DomaineMath* domain = new DomaineMath(-2.1, -1.3, 0.8, 1.3);
  Mandelbrot* algo = new Mandelbrot();

  return new FractaleMOO(960, 960, domain, algo, 30, 100);
}
