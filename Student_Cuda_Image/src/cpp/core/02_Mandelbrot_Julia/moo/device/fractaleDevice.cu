#include "Indice2D.h"
#include "IndiceTools.h"
#include "Mandelbrot.h"
#include "Julia.h"
#include "DomaineMath.h"

__global__ void processMandelbrot(uchar4* ptrDevPixels, int w, int h, int n, const DomaineMath& domaineMath);
__global__ void processJulia(uchar4* ptrDevPixels, int w, int h, int n, float c1, float c2, const DomaineMath& domaineMath);

__global__ void processMandelbrot(uchar4* ptrDevPixels, int w, int h, int n, const DomaineMath& domaineMath) {
	const int WH=w*h;
	const int NB_THREADS = Indice2D::nbThread();
	const int TID = Indice2D::tid();
	int s = TID;

	int i, j, nmax;
	double x, y;

	Mandelbrot algo;

	while (s < WH) {

		IndiceTools::toIJ(s, w, &i, &j); // s[0,W*H[ --> i[0,H[ j[0,W[
  	domaineMath.toXY(i, j, &x, &y );

		algo.checkSuit(x, y, n, &nmax);

		s += NB_THREADS;
	}
}

__global__ void processJulia(uchar4* ptrDevPixels, int w, int h, int n, float c1, float c2, const DomaineMath& domaineMath) {
	const int WH=w*h;
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();

	int i, j;
  double x, y;

	Julia algo(c1, c2);

  int s = TID;
	while (s < WH) {

		IndiceTools::toIJ(s, w, &i, &j); // s[0,W*H[ --> i[0,H[ j[0,W[
  	domaineMath.toXY(i, j, &x, &y );

		// algo.colorXY(&ptrDevPixels[s], algo.checkSuit(x, y, n), n);

		s += NB_THREADS;
	}
}
