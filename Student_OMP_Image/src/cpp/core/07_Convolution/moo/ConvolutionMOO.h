#ifndef CONVOLUTION_MOO_H_
#define CONVOLUTION_MOO_H_

#include "Animable_I.h"
#include "CVCaptureVideo.h"

class ConvolutionMOO: public Animable_I {

  public:
    ConvolutionMOO(string videoPath, int kernelWidth, int kernelHeight, float* ptrKernel);
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

    virtual void setParallelPatern(ParallelPatern parallelPatern);

  private:
    void convolution(uchar4* ptrDevPixels, int imageWidth, int imageHeight, float* ptrDevKernel, int kernelWidth, int kernelHeight);
    void convertInBlackAndWhite(uchar4* ptrDevPixels, int imageWidth, int imageHeight);
    void computeMinMax(uchar4* ptrPixels, int imageSize, int* ptrMin, int* ptrMax);
    
    // Inputs
    int t, kernelWidth, kernelHeight;
    float* ptrKernel;
    CVCaptureVideo* videoCapter;
};

#endif
