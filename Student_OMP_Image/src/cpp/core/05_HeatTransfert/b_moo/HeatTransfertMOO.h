#ifndef HEAT_TRANSFERT_MOO_H_
#define HEAT_TRANSFERT_MOO_H_

#include "cudaType.h"
#include "Animable_I.h"

class HeatTransfertMOO: public Animable_I {

  public:
  	HeatTransfertMOO(unsigned int w, unsigned int h, float* ptrImageInit, float* ptrImageHeater);
  	virtual ~HeatTransfertMOO();

  	virtual void process(uchar4* ptrPixels, int w, int h);
  	virtual void animationStep();

  	virtual float getAnimationPara();
  	virtual int getW();
  	virtual int getH();
  	virtual string getTitle();

  	virtual void setParallelPatern(ParallelPatern parallelPatern);

  private:
    void diffuse(float* ptrImageInput, float* ptrImageOutput);
    void crush(float* ptrImageHeater, float* ptrImage);
    void toScreen(float* ptrImage, uchar4* ptrPixels);

    // Inputs
  	unsigned int w;
  	unsigned int h;
    unsigned int wh;

    // Images
    float* ptrImageInit;
    float* ptrImageHeater;
    float* ptrImageA;
    float* ptrImageB;

  	// Tools
  	ParallelPatern parallelPatern;
    unsigned int iteration;
    unsigned int nbThreads;
};

#endif
