#include <iostream>
#include <omp.h>

#include "FractaleMOO.h"
#include "OmpTools.h"
#include "IndiceTools.h"

using std::cout;
using std::endl;
using std::string;

const unsigned int FractaleMOO::NB_THREADS = OmpTools::setAndGetNaturalGranularity();

FractaleMOO::FractaleMOO(int w, int h, DomaineMath* domain, Fractale* algo, int nmin, int nmax) {
	this->algo = algo;
	this->domain = domain;
  this->nmin = nmin;
  this->nmax = nmax;
  this->w = w;
  this->h = h;
  this->n = this->nmin;
	this->step = 1;
  this->parallelPatern = OMP_MIXTE;
}

FractaleMOO::~FractaleMOO() {
  delete this->algo;
	delete this->domain;
}

/**
 * Override
 */
void FractaleMOO::process( uchar4* ptrTabPixels, int w, int h, const DomaineMath& domaineMath ) {


	switch (parallelPatern) {

		case OMP_ENTRELACEMENT: // Plus lent sur CPU
	  {
		  entrelacementOMP(ptrTabPixels, w, h, domaineMath);
		  break;
	  }

		case OMP_FORAUTO: // Plus rapide sur CPU
	  {
		  forAutoOMP(ptrTabPixels, w, h, domaineMath);
		  break;
	  }

		case OMP_MIXTE: // Pour tester que les deux implementations fonctionnent
		{
		  // Note : Des saccades peuvent apparaitre � cause de la grande difference de fps entre la version entrelacer et auto
		  static bool isEntrelacement = true;

		  if (isEntrelacement) {
				entrelacementOMP(ptrTabPixels, w, h, domaineMath);
			} else {
				forAutoOMP(ptrTabPixels, w, h, domaineMath);
			}

		  isEntrelacement = !isEntrelacement; // Pour swithcer a chaque iteration
		  break;
	  }
	}
}

DomaineMath* FractaleMOO::getDomaineMathInit() {
	return this->domain;
}

/**
 * Override
 */
void FractaleMOO::animationStep() {

	if( this->n == this->nmax ) {
		this->step = -1;
	} else if(this->n == this->nmin ) {
		this->step = 1;
	}

	this->n += this->step;
}

/**
 * Override
 */
float FractaleMOO::getAnimationPara() {
	return (float) this->n;
}

/**
 * Override
 */
int FractaleMOO::getW()	{
	return this->w;
}

/**
 * Override
 */
int FractaleMOO::getH() {
	return this->h;
}

/**
 * Override
 */
string FractaleMOO::getTitle() {
	return "Fractale_" + this->algo->getName() + "_OMP";
}

void FractaleMOO::setParallelPatern(ParallelPatern parallelPatern) {
	this->parallelPatern = parallelPatern;
}

/**
 * Code entrainement Cuda
 */
void FractaleMOO::entrelacementOMP(uchar4* ptrTabPixels, int w, int h, const DomaineMath& domaineMath) {
	const int WH=w*h;

	#pragma omp parallel
	{
		const int TID = OmpTools::getTid();
		int s = TID;

		int i;
		int j;

		while (s < WH) {
			IndiceTools::toIJ(s, w, &i, &j); // s[0,W*H[ --> i[0,H[ j[0,W[

			this->workPixel(&ptrTabPixels[s], i, j, domaineMath);

			s += FractaleMOO::NB_THREADS;
		}
	}
}

/**
 * Code naturel et direct OMP
 */
void FractaleMOO::forAutoOMP(uchar4* ptrTabPixels, int w, int h, const DomaineMath& domaineMath) {

	#pragma omp parallel for
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++) {
			//int s = i * W + j;
			int s=IndiceTools::toS(w, i, j);// i[0,H[ j[0,W[  --> s[0,W*H[

			workPixel(&ptrTabPixels[s], i, j, domaineMath);
		}
	}
}

void FractaleMOO::workPixel(uchar4* ptrColorIJ, int i, int j, const DomaineMath& domaineMath) {
	double x, y;
	domaineMath.toXY(i, j, &x, &y );
	this->algo->colorXY(ptrColorIJ, x, y, this->n );
}
