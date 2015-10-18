#ifndef FRACTALE_H_
#define FRACTALE_H_

#include "ColorTools.h"
#include "cudaType.h"

class Fractale {
  public:
  	virtual ~Fractale();

  	void colorXY(uchar4* ptrColor, float x, float y, int n);

  	virtual int checkSuit(float z, float y, int n)=0;
  	virtual bool isDivergent(float z)=0;
};
#endif
