#ifndef HEAT_TRANSFERT_MOO_H_
#define HEAT_TRANSFERT_MOO_H_

#include "cudaType.h"
#include "Animable_I.h"

class HeatTransfertMOO: public Animable_I {
  public:

  	HeatTransfertMOO(unsigned int w, unsigned int h, float* imageInit, float* heaters);
  	virtual ~HeatTransfertMOO();

  	virtual void process(uchar4* ptrDevPixels, int w, int h);
  	virtual void animationStep();

  	virtual float getAnimationPara();
  	virtual int getW();
  	virtual int getH();
  	virtual string getTitle();

  	virtual void setParallelPatern(ParallelPatern parallelPatern);

    private:

  	void entrelacementOMP(uchar4* ptrTabPixels, int w, int h); 	// Code entrainement Cuda
  	void forAutoOMP(uchar4* ptrTabPixels, int w, int h); 		// Code naturel et direct OMP, plus performant


  private:

    // Inputs
  	unsigned int w;
  	unsigned int h;

    // Images
    float* imageInit;
    float* imageHeater;
    float* imageA;
    float* imageB;

  	// Tools
  	ParallelPatern parallelPatern;
};

#endif
