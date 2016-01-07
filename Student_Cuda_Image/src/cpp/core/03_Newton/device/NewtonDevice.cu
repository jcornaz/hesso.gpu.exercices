#include "Indice2D.h"
#include "IndiceTools.h"
#include "DomaineMath.h"
#include "NewtonMath.h"

__global__ void newton(uchar4* ptrDevTabPixels, int w, int h, int n, DomaineMath* ptrDevDomain, NewtonMath* ptrDevMath);

__global__ void newton(uchar4* ptrDevTabPixels, int w, int h, int n, DomaineMath* ptrDevDomain, NewtonMath* ptrDevMath) {
	const int NB_THREADS = Indice2D::nbThread();
	const int TID = Indice2D::tid();

	const int WH=w*h;

	int i, j;
  double x, y;

	int s = TID;
	while (s < WH) {

		IndiceTools::toIJ(s, w, &i, &j);
    ptrDevDomain->toXY(i, j, &x, &y );
    ptrDevMath->colorXY(&ptrDevTabPixels[s], x, y, n);

		s += NB_THREADS;
	}
}
