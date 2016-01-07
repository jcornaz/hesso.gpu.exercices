#include "RayTracingProvider.h"

RayTracing* RayTracingProvider::createMOO() {

  int dw = 512;
  int dh = 512;

  float dt = 0.1;
  int padding = 10;
  int nbSpheres = 100;

  return new RayTracing(dw, dh, padding, dt, nbSpheres);
}

Image* RayTracingProvider::createGL() {
  return new Image(createMOO());
}
