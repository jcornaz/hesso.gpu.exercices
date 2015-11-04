#include "cudaType.h"
#include "Sphere.h"
#include "OmpTools.h"
#include "IndiceTools.h"
#include "DomaineMath.h"
#include "ColorTools.h"

#define MAX_DISTANCE 1e80

void raytracing(uchar4* ptrDevPixels, int w, int h, Sphere** ptrDevSpheres, int nbSheres, float t);
void computeNearestSphere(Sphere** ptrDevSpheres, int nbSpheres, float2 floorPoint, int* nearestSphereIndex, float* brightness);

void raytracing(uchar4* ptrDevPixels, int w, int h, Sphere** ptrDevSpheres, int nbSpheres, float t) {
  const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();
  const int WH = w * h;

  #pragma omp parallel
  {
    const int TID = OmpTools::getTid();

    int i, j;
    Sphere* ptrDevNearestSphere;
    float2 floorPoint;
    float3 hsb;
    int nearestSphereIndex;
    float brightness;

    int s = TID;
    while (s < WH) {
      IndiceTools::toIJ(s, w, &i, &j);

      floorPoint.x = (float) j;
      floorPoint.y = (float) i;

      ptrDevNearestSphere = NULL;

      computeNearestSphere(ptrDevSpheres, nbSpheres, floorPoint, &nearestSphereIndex, &brightness);

      if (nearestSphereIndex < 0) {
        hsb.x = 0;
        hsb.y = 0;
        hsb.z = 0;
      } else {
        hsb.x = ptrDevSpheres[nearestSphereIndex]->hue(t);
        hsb.y = 1;
        hsb.z = brightness;
      }

      ColorTools::HSB_TO_RVB(hsb, &ptrDevPixels[s]);
      ptrDevPixels[s].w = 255;

      s += NB_THREADS;
    }
  }
}

void computeNearestSphere(Sphere** ptrDevSpheres, int nbSpheres, float2 floorPoint, int* nearestSphereIndex, float* brightness) {
  float hCarre, dz, distance;
  float distanceMin = MAX_DISTANCE;
  *nearestSphereIndex = -1;

  for( int i = 0 ; i < nbSpheres ; i++ ) {

    hCarre = ptrDevSpheres[i]->hCarre(floorPoint);

    if (ptrDevSpheres[i]->isEnDessous(hCarre)) {
      dz = ptrDevSpheres[i]->dz(hCarre);
      distance = ptrDevSpheres[i]->distance(dz);

      if (distance < distanceMin) {
        distanceMin = distance;
        *nearestSphereIndex = i;
        *brightness = ptrDevSpheres[i]->brightness(dz);
      }
    }
  }
}
