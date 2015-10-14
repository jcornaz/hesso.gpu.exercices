#include "Fractale.h"
#include "ColorTools.h"

Fractale::~Fractale() {
    // Nothing
}

void Fractale::colorXY(uchar4* ptrColor, float x, float y, int n)
    {
    int nmax = this->checkSuit(x, y, n);

    float hue;

    if (nmax > n)
	{
	hue = 1.0;
	}
    else
	{
	hue = ((float) nmax) / ((float) n);
	}

    ColorTools::HSB_TO_RVB(hue, ptrColor);
    ptrColor->w = 255;
    }
