#ifndef RAY_TRACING_H_
#define RAY_TRACING_H_

#include "Animable_I.h"
#include "MathTools.h"

class RayTracing: public Animable_I {

  public:
    RayTracing(int w, int h, float dt);
    virtual ~RayTracing();

    /**
    * Call periodicly by the api
    */
    virtual void process(uchar4* ptrDevPixels, int w, int h);

    /**
    * Call periodicly by the api
    */
    virtual void animationStep();

    virtual void setParallelPatern(ParallelPatern parallelPatern);

    virtual float getAnimationPara();
    virtual string getTitle();
    virtual int getW();
    virtual int getH();

  private:
    // Inputs
    int w;
    int h;
    float dt;

    // Tools
    float t;

    //Outputs
    string title;
};

#endif
