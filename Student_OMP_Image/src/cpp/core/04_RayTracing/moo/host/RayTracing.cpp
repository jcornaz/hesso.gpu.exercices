#include <cstdlib>
#include "RayTracing.h"

extern void raytracing(uchar4* ptrDevPixels, int w, int h, Sphere** ptrDevSpheres, int nbSheres, float t);

RayTracing::RayTracing(int w, int h, int padding, float dt, int nbSpheres) {
  this->title = "Ray Tracing OMP";
  this->w = w;
  this->h = h;
  this->t = 0;
  this->dt = dt;

  this->nbSpheres = nbSpheres;
  this->spheres = new Sphere*[nbSpheres];

  const float R_MIN = 20.0;
  const float R_MAX = w / 10.0;
  const float X_MIN = R_MAX + padding;
  const float X_MAX = w - R_MAX - padding;
  const float Y_MIN = R_MAX + padding;
  const float Y_MAX = h - R_MAX - padding;
  const float Z_MIN = 10.0;
  const float Z_MAX = 2.0 * w;
  const float H_MIN = 0.0;
  const float H_MAX = 1.0;

  float3 center;
  float rayon, hue;
  for (int i = 0 ; i < this->nbSpheres ; i++ ) {

    center.x = X_MIN + rand() * (X_MAX - X_MIN) / (RAND_MAX + 1.0);
    center.y = Y_MIN + rand() * (Y_MAX - Y_MIN) / (RAND_MAX + 1.0);
    center.z = Z_MIN + rand() * (Z_MAX - Z_MIN) / (RAND_MAX + 1.0);

    rayon = R_MIN + rand() * (R_MAX - R_MIN) / (RAND_MAX + 1.0);
    hue = H_MIN + rand() * (H_MAX - H_MIN) / (RAND_MAX);

    std::cout << center.x << ", " << center.y << ", " << center.z << " => " << rayon << ", " << hue << std::endl;

    this->spheres[i] = new Sphere(center, rayon, hue);
  }
}

RayTracing::~RayTracing() {
  for (int i = 0 ; i < this->nbSpheres ; i++ ) {
    delete this->spheres[i];
  }
}

/**
* Call periodicly by the api
*/
void RayTracing::process(uchar4* ptrDevPixels, int w, int h) {
  raytracing(ptrDevPixels, this->w, this->h, this->spheres, this->nbSpheres, this->t);
}

/**
* Call periodicly by the api
*/
void RayTracing::animationStep() {
  this->t += this->dt;
}

void RayTracing::setParallelPatern(ParallelPatern parallelPatern) {
  // Noghing this class use only the entrelacement version
}

float RayTracing::getAnimationPara() {
  return this->t;
}

string RayTracing::getTitle() {
  return this->title;;
}

int RayTracing::getW() {
  return this->w;
}

int RayTracing::getH() {
  return this->h;
}
