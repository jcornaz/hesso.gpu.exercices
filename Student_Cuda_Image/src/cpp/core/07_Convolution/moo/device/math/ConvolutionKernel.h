#ifndef CONVOLUTION_KERNEL_H_
#define CONVOLUTION_KERNEL_H_


class ConvolutionKernel {

  public:
    ConvolutionKernel(int w, int h, float* weights) {
      this->w = w;
      this->h = h;
      this->weights = weights;
    }

    int getW() {
      return this->w;
    }

    int getH() {
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
