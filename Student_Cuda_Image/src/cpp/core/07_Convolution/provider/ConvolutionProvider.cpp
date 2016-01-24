#include "ConvolutionProvider.h"

ConvolutionMOO* ConvolutionProvider::createMOO() {
  return ConvolutionProvider::createMOO(4096, 1024);
}

ConvolutionMOO* ConvolutionProvider::createMOO(int cudaGridDim, int cudaBlockDim) {
  float weights[81] = { 0.0828, 0.1987, 0.3705, 0.5366, 0.6063, 0.5366, 0.3705, 0.1987, 0.0828, 0.1987, 0.4746, 0.8646, 1.1794, 1.2765, 1.1794, 0.8646, 0.4746, 0.1987, 0.3705, 0.8646, 1.3475, 1.0033, 0.4061, 1.0033, 1.3475, 0.8646, 0.3705, 0.5366, 1.1794, 1.0033, -2.8306, -6.4829, -2.8306, 1.0033, 1.1794, 0.5366, 0.6063, 1.2765, 0.4061, -6.4829, -12.7462, -6.4829, 0.4061, 1.2765, 0.6063, 0.5366, 1.1794, 1.0033, -2.8306, -6.4829, -2.8306, 1.0033, 1.1794, 0.5366, 0.3705, 0.8646, 1.3475, 1.0033, 0.4061, 1.0033, 1.3475, 0.8646, 0.3705, 0.1987, 0.4746, 0.8646, 1.1794, 1.2765, 1.1794, 0.8646, 0.4746, 0.1987, 0.0828, 0.1987, 0.3705, 0.5366, 0.6063, 0.5366, 0.3705, 0.1987, 0.0828};

  for (int i = 0 ; i < 81 ; i++) {
    weights[i] = weights[i] / 100.0;
  }

  return new ConvolutionMOO("/media/Data/Video/autoroute.mp4", weights, cudaGridDim, cudaBlockDim);
}

Image* ConvolutionProvider::createGL() {
  return new Image(createMOO());
}
