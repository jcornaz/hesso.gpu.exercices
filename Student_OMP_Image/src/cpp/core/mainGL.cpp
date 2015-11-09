#include <iostream>

#include "GLUTImageViewers.h"
#include "Settings.h"

#include "ViewerZoomable.h"
#include "Viewer.h"

#include "RipplingProvider.h"
#include "MandelbrotProvider.h"
#include "JuliaProvider.h"
#include "RayTracingProvider.h"
#include "NewtonProvider.h"
#include "HeatTransfertProvider.h"

using std::cout;
using std::endl;

int mainGL(Settings& settings);

int mainGL(Settings& settings) {
  cout << "\n[OpenGL] mode" << endl;

  GLUTImageViewers::init(settings.getArgc(), settings.getArgv());

  Viewer<RipplingProvider> rippling(0, 0);
  ViewerZoomable<MandelbrotProvider> mandelbrot(300, 0);
  ViewerZoomable<JuliaProvider> julia(600, 0);
  ViewerZoomable<NewtonProvider> newton(0, 300);
  Viewer<RayTracingProvider> raytracing(300, 0);
  Viewer<HeatTransfertProvider> heatTransfert(600, 0);

  GLUTImageViewers::runALL();  // Bloquant, Tant qu'une fenetre est ouverte

  return EXIT_SUCCESS;
}
