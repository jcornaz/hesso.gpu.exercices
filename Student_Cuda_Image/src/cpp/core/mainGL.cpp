#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "GLUTImageViewers.h"
#include "Option.h"
#include "Viewer.h"
#include "ViewerZoomable.h"

#include "MandelbrotProvider.h"
#include "JuliaProvider.h"

using std::cout;
using std::endl;
using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainGL(Option& option);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainGL(Option& option)
    {
    cout << "\n[OpenGL] mode" << endl;

    GLUTImageViewers::init(option.getArgc(), option.getArgv());

    // Viewer : (int,int,boolean) : (px,py,isAnimation=true)
    Viewer<MandelbrotProvider> mandelbrot(0, 0);
    Viewer<JuliaProvider> julia(960, 0);
    // add here other viewer

    GLUTImageViewers::runALL(); // Bloquant, Tant qu'une fenetre est ouverte

    return EXIT_SUCCESS;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
