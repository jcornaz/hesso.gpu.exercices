#include <iostream>
#include <omp.h>

#include "NewtonMOO.h"
#include "OmpTools.h"
#include "IndiceTools.h"

const unsigned int NewtonMOO::NB_THREADS = OmpTools::setAndGetNaturalGranularity();

NewtonMOO::NewtonMOO(int w, int h, DomaineMath* domain) {
	this->domain = domain;
  this->w = w;
  this->h = h;
  this->parallelPatern = OMP_MIXTE;
}

NewtonMOO::~NewtonMOO() {
	delete this->domain;
}

/**
 * Override
 */
void NewtonMOO::process( uchar4* ptrTabPixels, int w, int h, const DomaineMath& domaineMath ) {


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

DomaineMath* NewtonMOO::getDomaineMathInit() {
	return this->domain;
}

/**
 * Override
 */
void NewtonMOO::animationStep() {
	// TODO
}

/**
 * Override
 */
float NewtonMOO::getAnimationPara() {
	return 0.0; // TODO
}

/**
 * Override
 */
int NewtonMOO::getW()	{
	return this->w;
}

/**
 * Override
 */
int NewtonMOO::getH() {
	return this->h;
}

/**
 * Override
 */
string NewtonMOO::getTitle() {
	return "Fractale Newton";
}

void NewtonMOO::setParallelPatern(ParallelPatern parallelPatern) {
	this->parallelPatern = parallelPatern;
}

/**
 * Code entrainement Cuda
 */
void NewtonMOO::entrelacementOMP(uchar4* ptrTabPixels, int w, int h, const DomaineMath& domaineMath) {
	const int WH=w*h;

	#pragma omp parallel
	{
		const int TID = OmpTools::getTid();

		int i;
		int j;

		int s = TID;
		while (s < WH) {
			IndiceTools::toIJ(s, w, &i, &j); // s[0,W*H[ --> i[0,H[ j[0,W[

			this->workPixel(&ptrTabPixels[s], i, j, domaineMath);

			s += NewtonMOO::NB_THREADS;
		}
	}
}

/**
 * Code naturel et direct OMP
 */
void NewtonMOO::forAutoOMP(uchar4* ptrTabPixels, int w, int h, const DomaineMath& domaineMath) {

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

void NewtonMOO::workPixel(uchar4* ptrColorIJ, int i, int j, const DomaineMath& domaineMath) {
	double x, y;
	domaineMath.toXY(i, j, &x, &y );
	// TODO
}
