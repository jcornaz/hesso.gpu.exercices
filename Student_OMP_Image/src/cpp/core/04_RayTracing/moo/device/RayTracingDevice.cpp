#include "cudaType.h"
#include "Sphere.h"
#include "OmpTools.h"
#include "IndiceTools.h"
#include "DomaineMath.h"
#include "ColorTools.h"

#define MAX_DISTANCE 1e80

void raytracing(uchar4* ptrDevPixels, int w, int h, Sphere** ptrDevSpheres, int nbSheres, float t);
void computeNearestSphere(Sphere** ptrDevSpheres, int nbSheres, float2 floorPoint, Sphere** nearest, float* brightness);

void raytracing(uchar4* ptrDevPixels, int w, int h, Sphere** ptrDevSpheres, int nbSpheres, float t) {
  const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();
  const int WH = w * h;

  #pragma omp parallel
  {
    const int TID = OmpTools::getTid();

    int i, j;
    Sphere* nearestSphere;
    float2 floorPoint;
    float3 hsb;
    float brightness;

    int s = TID;
    while (s < WH) {
      IndiceTools::toIJ(s, w, &i, &j);

      floorPoint.x = (float) j;
      floorPoint.y = (float) i;

      nearestSphere = NULL;

      computeNearestSphere(ptrDevSpheres, nbSpheres, floorPoint, &nearestSphere, &brightness);

      if (nearestSphere == NULL) {
        hsb.x = 0;
        hsb.y = 0;
        hsb.z = 0;
      } else {
        hsb.x = nearestSphere->hue(t);
        hsb.y = 1;
        hsb.z = brightness;
      }

      ColorTools::HSB_TO_RVB(hsb, &ptrDevPixels[s]);
      ptrDevPixels[s].w = 255;

      s += NB_THREADS;
    }
  }
}

void computeNearestSphere(Sphere** ptrDevSpheres, int nbSpheres, float2 floorPoint, Sphere** nearest, float* brightness) {

  float hCarre, dz, distance;
  float distanceMin = MAX_DISTANCE;

  int s = 0;
  while (s < nbSpheres) {
    hCarre = ptrDevSpheres[s]->hCarre(floorPoint);

    if (ptrDevSpheres[s]->isEnDessous(hCarre)) {
      dz = ptrDevSpheres[s]->dz(hCarre);
      distance = ptrDevSpheres[s]->distance(dz);

      if (distance < distanceMin) {
        distanceMin = distance;
        *nearest = ptrDevSpheres[s];
        *brightness = ptrDevSpheres[s]->brightness(dz);
      }
    }
  }
}
