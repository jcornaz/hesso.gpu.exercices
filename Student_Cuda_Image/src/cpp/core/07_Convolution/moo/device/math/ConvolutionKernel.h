#ifndef CONVOLUTION_KERNEL_H_
#define CONVOLUTION_KERNEL_H_


class ConvolutionKernel {

  public:
    ConvolutionKernel(int w, int h, float* weights) {
      this->w = w;
      this->h = h;
      this->weights = weights;
    }

    int getWidth() {
      return this->w;
    }

    int getHeight() {
      return this->h;
    }

    int getSize() {
      return this->w * this->h;
    }

    float* getWeights() {
      return this->weights;
    }

  private:
    int w;
    int h;
    float* weights;
};

#endif
