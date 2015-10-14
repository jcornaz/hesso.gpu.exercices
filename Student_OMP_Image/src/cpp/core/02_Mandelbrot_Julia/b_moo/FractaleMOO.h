#ifndef FRACTALE_MOO_H_
#define FRACTALE_MOO_H_

#include "cudaType.h"
#include "Animable_I.h"
#include "MathTools.h"
#include "Fractale.h"
#include "DomaineMath.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class FractaleMOO: public Animable_I
    {

	/*--------------------------------------*\
	|*		Constructeur		*|
	 \*-------------------------------------*/

    public:

	FractaleMOO(int w, int h, DomaineMath& domain, Fractale& algo, int nmin, int nmax);
	virtual ~FractaleMOO(void);

	/*--------------------------------------*\
	|*		Methode			*|
	 \*-------------------------------------*/

    public:

	/*--------------------------------------*\
	|*	Override Animable_I		*|
	 \*-----virtual--------------------------------*/

	virtual void process(uchar4* ptrDevImageGL, int w, int h);
	virtual void animationStep();

	virtual float getAnimationPara();
	virtual int getW();
	virtual int getH();
	virtual string getTitle();

	virtual void setParallelPatern(ParallelPatern parallelPatern);

    private:

	void entrelacementOMP(uchar4* ptrTabPixels,int w,int h); 	// Code entrainement Cuda
	void forAutoOMP(uchar4* ptrTabPixels,int w,int h); 		// Code naturel et direct OMP, plus performsnt
	void workPixel(uchar4* ptrColorIJ, int i, int j);
	/*--------------------------------------*\
	|*		Attribut		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	unsigned int w;
	unsigned int h;
	unsigned int nmin;
	unsigned int nmax;
	DomaineMath domain;
	Fractale& algo;

	// Tools
	unsigned int n;
	ParallelPatern parallelPatern;
    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
