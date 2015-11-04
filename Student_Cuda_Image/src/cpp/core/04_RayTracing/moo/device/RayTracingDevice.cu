#include "cudaType.h"

#include "Sphere.h"
#include "Indice2D.h"
#include "IndiceTools.h"
#include "ColorTools.h"

#define MAX_DISTANCE 1e80

__global__ void raytracing(uchar4* ptrDevPixels, int w, int h, Sphere** ptrDevSpheres, int nbSheres, float t);
__device__ void computeNearestSphere(Sphere** ptrDevSpheres, int nbSpheres, float2 floorPoint, int* nearestSphereIndex, float* brightness);

__global__ void raytracing(uchar4* ptrDevPixels, int w, int h, Sphere** ptrDevSpheres, int nbSpheres, float t) {
  const int NB_THREADS = Indice2D::nbThread();
	const int TID = Indice2D::tid();
  const int WH = w * h;

  int i, j;
  float2 floorPoint;
  float3 hsb;
  int nearestSphereIndex;
  float dz;

  int s = TID;
  while (s < WH) {
    IndiceTools::toIJ(s, w, &i, &j);

    floorPoint.x = (float) j;
    floorPoint.y = (float) i;

    computeNearestSphere(ptrDevSpheres, nbSpheres, floorPoint, &nearestSphereIndex, &dz);

    if (nearestSphereIndex < 0) {
      hsb.x = 0;
      hsb.y = 0;
      hsb.z = 0;
    } else {
      hsb.x = ptrDevSpheres[nearestSphereIndex]->hue(t);
      hsb.y = 1;
      hsb.z = ptrDevSpheres[nearestSphereIndex]->brightness(dz);;
    }

    ColorTools::HSB_TO_RVB(hsb, &ptrDevPixels[s]);
    ptrDevPixels[s].w = 255;

    s += NB_THREADS;
  }
}

__device__ void computeNearestSphere(Sphere** ptrDevSpheres, int nbSpheres, float2 floorPoint, int* nearestSphereIndex, float* dz) {
  float hCarre, currentDz, distance;
  float distanceMin = MAX_DISTANCE;
  *nearestSphereIndex = -1;

  for( int i = 0 ; i < nbSpheres ; i++ ) {

    hCarre = ptrDevSpheres[i]->hCarre(floorPoint);

    if (ptrDevSpheres[i]->isEnDessous(hCarre)) {
      currentDz = ptrDevSpheres[i]->dz(hCarre);
      distance = ptrDevSpheres[i]->distance(currentDz);

      if (distance < distanceMin) {
        distanceMin = distance;
        *nearestSphereIndex = i;
        *dz = currentDz;
      }
    }
  }
}
