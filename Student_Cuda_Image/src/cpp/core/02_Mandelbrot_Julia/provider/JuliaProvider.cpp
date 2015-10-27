#include "JuliaProvider.h"
#include "FractaleMOO.h"
#include "Julia.h"
#include "DomaineMath.h"

ImageFonctionel* JuliaProvider::createGL() {
  AnimableFonctionel_I* ptrAnimable = JuliaProvider::createMOO();
  return new ImageFonctionel(ptrAnimable);
}

AnimableFonctionel_I* JuliaProvider::createMOO() {
  DomaineMath* domain = new DomaineMath(-1.3, -1.4, 1.3, 1.4);
  Julia* algo = new Julia(-0.12, 0.85);

  return new FractaleMOO(640, 640, domain, algo, 30, 50);
}
