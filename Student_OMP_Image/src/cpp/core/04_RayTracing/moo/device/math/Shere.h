#ifndef SPHERE_H
#define SPHERE_H
  #ifndef PI
    #define PI 3.141592653589793f
  #endif

class Sphere {

  public:

    Sphere(float3 centre, float rayon, float hue) {

      // Inputs
      this->centre = centre;
      this->r = rayon;
      this->hue = hue;

      // Tools
      this->rCarre = rayon * rayon;
    }

    /**
     * required by example for new Sphere[n]
     */
    Sphere() {
      // nothing
    }

    float hCarre(float2 xySol) {
      float a = (centre.x - xySol.x);
      float b = (centre.y - xySol.y);
      return a * a + b * b;
    }

    bool isEnDessous(float hCarre) {
      return hCarre < rCarre;
    }

    float dz(float hCarre) {
      return sqrtf(rCarre - hCarre);
    }

    float brightness(float dz) {
      return dz / r;
    }

    float distance(float dz) {
      return centre.z - dz;
    }

    float getHueStart() {
      return hueStart;
    }

    /**
     * usefull for animation
     */
    float hue(float t) {
      return 0.5 + 0.5 * sin(t + T + 3 * PI / 2);
    }

  private:

    // Inputs
    float r;
    float3 centre;
    float hueStart;

    // Tools
    float rCarre;
    float T ; // usefull for animation
}
