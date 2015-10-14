#include <iostream>
#include <omp.h>

#include "FractaleMOO.h"
#include "OmpTools.h"
#include "IndiceTools.h"
#include "../c_math/Fractale.h"
#include "DomaineMath.h"

using std::cout;
using std::endl;
using std::string;

FractaleMOO::FractaleMOO(int w, int h, DomaineMath& domain, Fractale& algo, int nmin, int nmax):
	algo(algo),
	domain(domain)
    {
    this->nmin = nmin;
    this->nmax = nmax;
    this->w = w;
    this->h = h;
    this->n = this->nmin;
    this->parallelPatern = OMP_MIXTE;
    }

FractaleMOO::~FractaleMOO(void)
    {
    // Nothing
    }

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * Override
 */
void FractaleMOO::process(uchar4* ptrTabPixels, int w, int h)
    {
    switch (parallelPatern)
	{

	case OMP_ENTRELACEMENT: // Plus lent sur CPU
	    {
	    entrelacementOMP(ptrTabPixels, w, h);
	    break;
	    }

	case OMP_FORAUTO: // Plus rapide sur CPU
	    {
	    forAutoOMP(ptrTabPixels, w, h);
	    break;
	    }

	case OMP_MIXTE: // Pour tester que les deux implementations fonctionnent
	    {
	    // Note : Des saccades peuvent apparaitre � cause de la grande difference de fps entre la version entrelacer et auto
	    static bool isEntrelacement = true;
	    if (isEntrelacement)
		{
		entrelacementOMP(ptrTabPixels, w, h);
		}
	    else
		{
		forAutoOMP(ptrTabPixels, w, h);
		}
	    isEntrelacement = !isEntrelacement; // Pour swithcer a chaque iteration
	    break;
	    }
	}
    }

/**
 * Override
 */
void FractaleMOO::animationStep()
    {
    this->n++;
    }

/*--------------*\
 |*	get	*|
 \*-------------*/

/**
 * Override
 */
float FractaleMOO::getAnimationPara()
    {
    return (float) this->n;
    }

/**
 * Override
 */
int FractaleMOO::getW()
    {
    return w;
    }

/**
 * Override
 */
int FractaleMOO::getH()
    {
    return h;
    }

/**
 * Override
 */
string FractaleMOO::getTitle()
    {
    return "Fractale_OMP";
    }

/*-------------*\
 |*     set	*|
 \*------------*/

void FractaleMOO::setParallelPatern(ParallelPatern parallelPatern)
    {
    this->parallelPatern = parallelPatern;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/**
 * Code entrainement Cuda
 */
void FractaleMOO::entrelacementOMP(uchar4* ptrTabPixels, int w, int h)
    {
    const int WH=w*h;

#pragma omp parallel
	{
	const int NB_THREAD = OmpTools::getNbThread();

	const int TID = OmpTools::getTid();
	int s = TID;

	int i;
	int j;
	while (s < WH)
	    {
	    IndiceTools::toIJ(s,w,&i,&j); // s[0,W*H[ --> i[0,H[ j[0,W[

	    this->workPixel(&ptrTabPixels[s],i, j);

	    s += NB_THREAD;
	    }
	}
    }

/**
 * Code naturel et direct OMP
 */
void FractaleMOO::forAutoOMP(uchar4* ptrTabPixels, int w, int h)
    {

#pragma omp parallel for
    for (int i = 0; i < h; i++)
	{
	for (int j = 0; j < w; j++)
	    {
	    //int s = i * W + j;
	    int s=IndiceTools::toS(w,i,j);// i[0,H[ j[0,W[  --> s[0,W*H[

	    workPixel(&ptrTabPixels[s], i, j);
	    }
	}
    }

void FractaleMOO::workPixel(uchar4* ptrColorIJ, int i, int j)
    {
	double x, y;

	this->domain.toXY(i, j, &x, &y );
	this->algo.colorXY(ptrColorIJ, x, y, this->n );
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
