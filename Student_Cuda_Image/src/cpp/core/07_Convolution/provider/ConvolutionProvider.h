#ifndef CONVOLUTION_PROVIDER_H_
#define CONVOLUTION_PROVIDER_H_

#include "Image.h"
#include "ConvolutionMOO.h"

class ConvolutionProvider {

  public:
    static ConvolutionMOO* createMOO();
  	static Image* createGL();
};

#endif
