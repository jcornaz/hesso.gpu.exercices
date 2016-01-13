#include "ConvolutionProvider.h"

ConvolutionMOO* ConvolutionProvider::createMOO() {
  ConvolutionKernel kernel;

  return new ConvolutionMOO(1920, 1080, kernel);
}

Image* ConvolutionProvider::createGL() {
  return new Image(createMOO());
}
