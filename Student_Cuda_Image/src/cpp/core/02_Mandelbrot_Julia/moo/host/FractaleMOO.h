#ifndef FRACTALE_MOO_H_
#define FRACTALE_MOO_H_

#include "cudaType.h"
#include "AnimableFonctionel_I.h"
#include "Fractale.h"
#include "DomaineMath.h"

class FractaleMOO: public AnimableFonctionel_I {

  public:
  	FractaleMOO(int w, int h, DomaineMath* domain, Fractale* algo, int nmin, int nmax);
  	virtual ~FractaleMOO(void);

  	virtual void process(uchar4* ptrDevPixels, int w, int h, const DomaineMath& domaineMath);
  	virtual void animationStep();

    virtual DomaineMath* getDomaineMathInit(void);
  	virtual float getAnimationPara();
  	virtual int getW();
  	virtual int getH();
  	virtual string getTitle();

  private:

  	// Inputs
  	unsigned int w;
  	unsigned int h;
  	unsigned int n;
  	unsigned int nmin;
  	unsigned int nmax;
    int step;
  	DomaineMath* ptrDevDomain;
    DomaineMath* ptrDomain;
  	Fractale* algo;

  	// Tools
  	dim3 dg;
  	dim3 db;
};

#endif
