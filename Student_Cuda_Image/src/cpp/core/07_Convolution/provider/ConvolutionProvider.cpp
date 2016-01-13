#include "ConvolutionProvider.h"

ConvolutionMOO* ConvolutionProvider::createMOO() {
  float weights[100];

  for (int i = 0 ; i < 100 ; i++) {
    weights[i] = 0.1f;
  }

  ConvolutionKernel kernel(10, 10, weights);

  return new ConvolutionMOO(1920, 1080, kernel);
}

Image* ConvolutionProvider::createGL() {
  return new Image(createMOO());
}
