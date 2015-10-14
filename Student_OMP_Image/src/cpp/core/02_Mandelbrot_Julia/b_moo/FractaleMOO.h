#ifndef FRACTALE_MOO_H_
#define FRACTALE_MOO_H_

#include "cudaType.h"
#include "AnimableFonctionel_I.h"
#include "MathTools.h"
#include "Fractale.h"
#include "DomaineMath.h"

class FractaleMOO: public AnimableFonctionel_I {
  public:
  	FractaleMOO(int w, int h, DomaineMath* domain, Fractale* algo, int nmin, int nmax);
  	virtual ~FractaleMOO(void);

  	virtual void process(uchar4* ptrDevPixels,int w, int h,const DomaineMath& domaineMath);
  	virtual void animationStep();

    virtual DomaineMath* getDomaineMathInit(void);
  	virtual float getAnimationPara();
  	virtual int getW();
  	virtual int getH();
  	virtual string getTitle();

  	virtual void setParallelPatern(ParallelPatern parallelPatern);

  private:
  	void entrelacementOMP(uchar4* ptrTabPixels,int w,int h, const DomaineMath& domaineMath); 	// Code entrainement Cuda
  	void forAutoOMP(uchar4* ptrTabPixels,int w,int h, const DomaineMath& domaineMath); 		// Code naturel et direct OMP, plus performsnt
  	void workPixel(uchar4* ptrColorIJ, int i, int j, const DomaineMath& domaineMath);


  	// Inputs
  	unsigned int w;
  	unsigned int h;
  	unsigned int nmin;
  	unsigned int nmax;
  	DomaineMath* domain;
  	Fractale* algo;

  	// Tools
  	unsigned int n;
  	ParallelPatern parallelPatern;
};

#endif
