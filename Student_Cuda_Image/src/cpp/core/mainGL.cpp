#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "GLUTImageViewers.h"
#include "Option.h"
#include "Viewer.h"
#include "ViewerZoomable.h"

#include "RipplingProvider.h"
#include "MandelbrotProvider.h"
#include "JuliaProvider.h"
#include "RayTracingProvider.h"

using std::cout;
using std::endl;
using std::string;

int mainGL(Option& option);

int mainGL(Option& option) {
	cout << "\n[OpenGL] mode" << endl;

	GLUTImageViewers::init(option.getArgc(), option.getArgv());

	Viewer<RipplingProvider> rippling(0, 0);
	// ViewerZoomable<MandelbrotProvider> mandelbrot(640, 0);
	// ViewerZoomable<JuliaProvider> julia(0, 640);
	Viewer<RayTracingProvider> raytracing(640, 640);

	GLUTImageViewers::runALL(); // Bloquant, Tant qu'une fenetre est ouverte

	return EXIT_SUCCESS;
}
