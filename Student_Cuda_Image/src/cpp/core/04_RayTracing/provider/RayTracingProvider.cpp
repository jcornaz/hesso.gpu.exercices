#include "RayTracingProvider.h"

RayTracing* RayTracingProvider::createMOO() {

  int dw = 1024;
  int dh = 1024;

  float dt = 0.1;
  int padding = 10;
  int nbSpheres = 100;

  return new RayTracing(dw, dh, padding, dt, nbSpheres);
}

Image* RayTracingProvider::createGL() {
  return new Image(createMOO());
}
