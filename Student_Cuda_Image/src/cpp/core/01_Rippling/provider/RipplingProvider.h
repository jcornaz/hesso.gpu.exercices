#ifndef RIPPLING_PROVIDER_H_
#define RIPPLING_PROVIDER_H_

#include "Rippling.h"
#include "Image.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class RipplingProvider
    {
    public:

	static Rippling* createMOO(void);
	static Image* createGL(void);

    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

