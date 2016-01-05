#include "NewtonProvider.h"
#include "NewtonMOO.h"
#include "DomaineMath.h"

ImageFonctionel* NewtonProvider::createGL() {
  AnimableFonctionel_I* ptrAnimable = NewtonProvider::createMOO();
  return new ImageFonctionel(ptrAnimable);
}

AnimableFonctionel_I* NewtonProvider::createMOO() {
  DomaineMath* domain = new DomaineMath(-2.0, -2.0, 2.0, 2.0);

  return new NewtonMOO(640, 640, domain);
}
