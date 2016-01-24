#ifndef CONVOLUTION_MOO_H_
#define CONVOLUTION_MOO_H_

#include "Animable_I.h"
#include "CVCaptureVideo.h"

class ConvolutionMOO: public Animable_I {

  public:
    ConvolutionMOO(string videoPath, int kernelWidth, float* ptrKernel);
    ~ConvolutionMOO();

    /**
    * Call periodicly by the api
    */
    virtual void process(uchar4* ptrDevPixels, int w, int h);

    /**
    * Call periodicly by the api
    */
    virtual void animationStep();
    virtual float getAnimationPara();
    virtual string getTitle();
    virtual int getW();
    virtual int getH();

  private:

    // Inputs
    int t, kernelWidth;
    float* ptrDevKernel;
    uchar4* ptrDevImage;
    CVCaptureVideo* videoCapter;

    // Tools
  	dim3 dg;
  	dim3 db;
};

#endif
