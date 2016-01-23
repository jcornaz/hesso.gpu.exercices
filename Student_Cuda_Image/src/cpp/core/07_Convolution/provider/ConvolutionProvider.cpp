#include "ConvolutionProvider.h"

ConvolutionMOO* ConvolutionProvider::createMOO() {
  float weights[2500];

  for (int i = 0 ; i < 2500 ; i++) {
    weights[i] = 1.0 / 2500.0;
  }

  return new ConvolutionMOO("/media/Data/Video/autoroute.mp4", 50, 50, weights);
}

Image* ConvolutionProvider::createGL() {
  return new Image(createMOO());
}
