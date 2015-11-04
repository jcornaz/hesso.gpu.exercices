#include "RayTracingProvider.h"

RayTracing* RayTracingProvider::createMOO() {

  int dw = 960;
  int dh = 960;

  float dt = 0.1;
  int padding = 10;
  int nbSpheres = 100;

  return new RayTracing(dw, dh, padding, dt, nbSpheres);
}

Image* RayTracingProvider::createGL() {
  return new Image(createMOO());
}
