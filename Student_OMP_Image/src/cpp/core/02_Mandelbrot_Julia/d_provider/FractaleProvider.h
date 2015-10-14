#ifndef FRACTALE_PROVIDER_H_
#define FRACTALE_PROVIDER_H_

#include "ImageFonctionel.h"
#include "AnimableFonctionel_I.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class FractaleProvider
    {
    public:

	static ImageFonctionel* createGL(void);
	static AnimableFonctionel_I* createMOO(void);

    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
