#include "MandelbrotMultiGPUProvider.h"
#include "MandelbrotMultiGPUMOO.h"
#include "Mandelbrot.h"
#include "DomaineMath.h"

ImageFonctionel* MandelbrotMultiGPUProvider::createGL() {
  AnimableFonctionel_I* ptrAnimable = MandelbrotMultiGPUProvider::createMOO();
  return new ImageFonctionel(ptrAnimable);
}

AnimableFonctionel_I* MandelbrotMultiGPUProvider::createMOO() {
  DomaineMath* domain = new DomaineMath(-2.1, -1.3, 0.8, 1.3);
  Mandelbrot* algo = new Mandelbrot();

  return new MandelbrotMultiGPUMOO(512, 512, domain, algo, 30, 50);
}
