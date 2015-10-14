#include <iostream>

#include "Indice2D.h"
#include "cudaTools.h"
#include "Device.h"
#include "IndiceTools.h"
#include "RipplingMath.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void rippling(uchar4* ptrDevPixels, int w, int h, float t);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

__global__ void rippling(uchar4* ptrDevPixels, int w, int h, float t) {
  RipplingMath ripplingMath = RipplingMath(w, h);

  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();
  const int n = w * h;
  int s = 0;
  while( s < n ) {
    int i, j;
    IndiceTools::toIJ(s, w, &i, &j);
    ripplingMath.colorIJ(&ptrDevPixels[s], i, j, t);
    s += NB_THREADS;
  }
}

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
