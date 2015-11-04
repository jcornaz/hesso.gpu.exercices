#ifndef SPHERE_H
#define SPHERE_H

#ifndef PI
  #define PI 3.141592653589793f
#endif

#include <cmath>

class Sphere {

  public:

    __device__ Sphere(float3 centre, float rayon, float hue) {

      // Inputs
      this->centre = centre;
      this->r = rayon;
      this->hueStart = hue;

      // Tools
      this->rCarre = rayon * rayon;
      this->T = 100.0;
    }

    /**
     * required by example for new Sphere[n]
     */
    __device__ Sphere() {
      // nothing
    }

    __device__ float hCarre(float2 xySol) {
      float a = (centre.x - xySol.x);
      float b = (centre.y - xySol.y);
      return a * a + b * b;
    }

    __device__ bool isEnDessous(float hCarre) {
      return hCarre < rCarre;
    }

    __device__ float dz(float hCarre) {
      return sqrtf(rCarre - hCarre);
    }

    __device__ float brightness(float dz) {
      return dz / r;
    }

    __device__ float distance(float dz) {
      return centre.z - dz;
    }

    /**
     * usefull for animation
     */
    __device__ float hue(float t) {
      return fmod(this->hueStart + t / this->T, 1.0);
    }

  private:

    // Inputs
    float r;
    float3 centre;
    float hueStart;

    // Tools
    float rCarre;
    float T;
};

#endif
