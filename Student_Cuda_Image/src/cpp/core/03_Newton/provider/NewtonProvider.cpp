#include "NewtonProvider.h"
#include "NewtonMOO.h"
#include "DomaineMath.h"
#include "NewtonMath.h"

ImageFonctionel* NewtonProvider::createGL() {
  AnimableFonctionel_I* ptrAnimable = NewtonProvider::createMOO();
  return new ImageFonctionel(ptrAnimable);
}

AnimableFonctionel_I* NewtonProvider::createMOO() {
  DomaineMath* domain = new DomaineMath(-2.0, -2.0, 2.0, 2.0);
  NewtonMath* math = new NewtonMath(0.001, 25);

  return new NewtonMOO(512, 512, domain, math);
}
