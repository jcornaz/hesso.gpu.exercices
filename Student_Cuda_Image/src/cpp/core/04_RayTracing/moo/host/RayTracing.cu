#include <cstdlib>

#include "cuda_runtime.h"
#include "RayTracing.h"
#include "Device.h"

__global__ void raytracing(uchar4* ptrDevPixels, int w, int h, Sphere** ptrDevSpheres, int nbSheres, float t);

RayTracing::RayTracing(int w, int h, int padding, float dt, int nbSpheres) {
  this->title = "Ray Tracing Cuda";
  this->w = w;
  this->h = h;
  this->t = 0;
  this->dt = dt;

  this->dg = dim3(8, 8, 1);
  this->db = dim3(16, 16, 1);

  this->padding = padding;

  this->createSpheres(nbSpheres);

  Device::assertDim(dg, db);
}

RayTracing::~RayTracing() {
  this->destroySpheres();
}

void RayTracing::createSpheres(int nb) {

  const float R_MIN = 20.0;
  const float R_MAX = this->w / 10.0;
  const float X_MIN = R_MAX + this->padding;
  const float X_MAX = this->w - R_MAX - this->padding;
  const float Y_MIN = R_MAX + this->padding;
  const float Y_MAX = this->h - R_MAX - this->padding;
  const float Z_MIN = 10.0;
  const float Z_MAX = 2.0 * w;
  const float H_MIN = 0.0;
  const float H_MAX = 1.0;

  float3 center;
  float rayon, hue;

  this->nbSpheres = nb;
  size_t sphereSize = sizeof(Sphere);
  size_t ptrSize = sizeof(Sphere*);

  Sphere* ptrCurrentSphere;
  Sphere* ptrDevCurrentSphere;

  HANDLE_ERROR(cudaMalloc(&this->ptrDevSpheres, ptrSize * nb));

  for (int i = 0 ; i < this->nbSpheres ; i++ ) {

    center.x = X_MIN + rand() * (X_MAX - X_MIN) / (RAND_MAX + 1.0);
    center.y = Y_MIN + rand() * (Y_MAX - Y_MIN) / (RAND_MAX + 1.0);
    center.z = Z_MIN + rand() * (Z_MAX - Z_MIN) / (RAND_MAX + 1.0);

    rayon = R_MIN + rand() * (R_MAX - R_MIN) / (RAND_MAX + 1.0);
    hue = H_MIN + rand() * (H_MAX - H_MIN) / (RAND_MAX);

    ptrCurrentSphere = new Sphere(center, rayon, hue);
    HANDLE_ERROR(cudaMalloc(&ptrDevCurrentSphere, sphereSize));
    HANDLE_ERROR(cudaMemcpy(ptrDevCurrentSphere, ptrCurrentSphere, sphereSize, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(&this->ptrDevSpheres[i], &ptrDevCurrentSphere, ptrSize, cudaMemcpyHostToDevice));
    delete ptrCurrentSphere;
  }
}

void RayTracing::destroySpheres() {

  for(int i = 0 ; i < this->nbSpheres ; i++)
  {
    HANDLE_ERROR(cudaFree(this->ptrDevSpheres[i]));
  }
  HANDLE_ERROR(cudaFree(this->ptrDevSpheres));

  this->nbSpheres = 0;
}

/**
* Call periodicly by the api
*/
void RayTracing::process(uchar4* ptrDevPixels, int w, int h) {
  raytracing<<<dg,db>>>(ptrDevPixels, this->w, this->h, this->ptrDevSpheres, this->nbSpheres, this->t);
}

/**
* Call periodicly by the api
*/
void RayTracing::animationStep() {
  this->t += this->dt;
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
