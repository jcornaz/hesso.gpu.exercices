#ifndef CUSTOM_MATH_TOOLS_H_
#define CUSTOM_MATH_TOOLS_H_

class CustomMathTools {

  public:

    /**
     * Fonction dont l'intégrale bornée de [0,1] vaut exactement PI
     * @param Valeur de x
     * @param Valeur de f(x)
     */
    __device__ static float fpi(float x) {
      return 4 / (1 + x * x);
    }
};

#endif
