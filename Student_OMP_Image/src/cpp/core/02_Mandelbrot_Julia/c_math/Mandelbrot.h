#ifndef MANDELBROT_H_
#define MANDELBROT_H_

#include "Fractale.h"

class Mandelbrot: public Fractale
    {
    public:
	virtual ~Mandelbrot();

	virtual int checkSuit(float z, float y, int n);
	virtual bool isDivergent(float z);
    };

#endif
