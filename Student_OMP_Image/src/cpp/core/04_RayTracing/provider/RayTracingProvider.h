#ifndef RAY_TRACING_PROVIDER_H_
#define RAY_TRACING_PROVIDER_H_

#include "RayTracing.h"
#include "Image.h"

class RayTracingProvider {
  
  public:
    static RayTracing* createMOO();
    static Image* createGL();
};

#endif
