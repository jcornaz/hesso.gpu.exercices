#include "Indice2D.h"
#include "IndiceTools.h"
#include "Mandelbrot.h"
#include "Julia.h"
#include "DomaineMath.h"

__device__ void processFractale(uchar4* ptrTabPixels, int w, int h, int n, const Fractale& algo, const DomaineMath& domaineMath) {
	const int WH=w*h;
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();

	int i, j;
  double x, y;

  int s = TID;
	while (s < WH) {
		IndiceTools::toIJ(s, w, &i, &j); // s[0,W*H[ --> i[0,H[ j[0,W[
  	domaineMath.toXY(i, j, &x, &y );

  	algo.colorXY(ptrTabPixels, x, y, n );

		s += NB_THREADS;
	}
}
