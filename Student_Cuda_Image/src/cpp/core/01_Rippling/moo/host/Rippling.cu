#include <iostream>
#include <assert.h>

#include "Rippling.h"
#include "Device.h"

using std::cout;
using std::endl;

extern __global__ void ripplingOneToOne(uchar4* ptrDevPixels, int w, int h, float t);
extern __global__ void ripplingOneDimension(uchar4* ptrDevPixels, int w, int h, float t);
extern __global__ void ripplingTwoDimensions(uchar4* ptrDevPixels, int w, int h, float t);

Rippling::Rippling(int w, int h, float dt) {
  assert(w == h);

  // Inputs
  this->w = w;
  this->h = h;
  this->dt = dt;

  // Tools

  this->dg = dim3(64, 64, 1);
  this->db = dim3(16, 16, 1);
  this->t = 0;

  // Outputs
  this->title = "Rippling_Cuda";

  //print(dg, db);
  Device::assertDim(dg, db);
}

Rippling::~Rippling() {
  // rien
}

/**
 * Override
 */
void Rippling::process(uchar4* ptrDevPixels, int w, int h) {
  ripplingOneToOne<<<dg,db>>>(ptrDevPixels, w, h, this->t);
  // ripplingOneDimension<<<dg,db>>>(ptrDevPixels, w, h, this->t);
  // ripplingTwoDimensions<<<dg,db>>>(ptrDevPixels, w, h, this->t);
}


/**
 * Override
 */
void Rippling::animationStep() {
  this->t += this->dt;
}

/**
 * Override
 */
float Rippling::getAnimationPara(void) {
  return t;
}

/**
 * Override
 */
int Rippling::getW(void) {
  return w;
}

/**
 * Override
 */
int Rippling::getH(void) {
  return  h;
}

/**
 * Override
 */
string Rippling::getTitle(void) {
  return title;
}
