#ifndef NEWTON_MOO_H_
#define NEWTON_MOO_H_

#include "cudaType.h"
#include "AnimableFonctionel_I.h"
#include "DomaineMath.h"
#include "NewtonMath.h"

class NewtonMOO: public AnimableFonctionel_I {

  public:
  	NewtonMOO(int w, int h, DomaineMath* ptrDomain, NewtonMath* ptrMath);
  	virtual ~NewtonMOO();

  	virtual void process(uchar4* ptrDevPixels,int w, int h,const DomaineMath& domaineMath);
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
  	DomaineMath* ptrDomain;
    DomaineMath* ptrDevDomain;
  	NewtonMath* ptrDevMath;

  	// Tools
    dim3 dg;
    dim3 db;
};

#endif
