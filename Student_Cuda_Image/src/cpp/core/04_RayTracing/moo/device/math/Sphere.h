#ifndef SPHERE_H
#define SPHERE_H

#ifndef PI
  #define PI 3.141592653589793f
#endif

#include <cmath>

class Sphere {

  public:

    Sphere(float3 centre, float rayon, float hue) {

      // Inputs
      this->centre = centre;
      this->r = rayon;
      this->hueStart = hue;

      // Tools
      this->rCarre = rayon * rayon;
      this->T = asin(2 * hue - 1) - 3 * PI / 2;
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
      return 0.5 + 0.5 * sin(t + 3 * PI / 2 + T);
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
