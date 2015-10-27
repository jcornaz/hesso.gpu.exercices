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

int mainGL(Option& option);

int mainGL(Option& option) {
	cout << "\n[OpenGL] mode" << endl;

	GLUTImageViewers::init(option.getArgc(), option.getArgv());

	ViewerZoomable<MandelbrotProvider> mandelbrot(0, 0);
	ViewerZoomable<JuliaProvider> julia(960, 0);

	GLUTImageViewers::runALL(); // Bloquant, Tant qu'une fenetre est ouverte

	return EXIT_SUCCESS;
}
