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
#include "NewtonProvider.h"
#include "RayTracingProvider.h"
#include "HeatTransfertProvider.h"

using std::cout;
using std::endl;
using std::string;

int mainGL(Option& option);

int mainGL(Option& option) {
	cout << "\n[OpenGL] mode" << endl;

	GLUTImageViewers::init(option.getArgc(), option.getArgv());

	Viewer<RipplingProvider> rippling(0, 0);
	ViewerZoomable<MandelbrotProvider> mandelbrot(512, 0);
	ViewerZoomable<JuliaProvider> julia(1024, 0);
	ViewerZoomable<NewtonProvider> newton(0, 512);
	Viewer<RayTracingProvider> raytracing(512, 512);
	Viewer<HeatTransfertProvider> heatTransfer(1024, 512);

	GLUTImageViewers::runALL(); // Bloquant, Tant qu'une fenetre est ouverte

	return EXIT_SUCCESS;
}
