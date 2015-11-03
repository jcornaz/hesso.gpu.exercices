#include "RayTracingProvider.h"

RayTracing* RayTracingProvider::createMOO() {
  float dt = 1;

  int dw = 960;
  int dh = 960;

  return new RayTracing(dw, dh, 100, dt, 50);
}

Image* RayTracingProvider::createGL() {
  return new Image(createMOO());
}
