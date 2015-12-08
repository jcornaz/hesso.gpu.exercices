#include <iostream>

#include "GLUTImageViewers.h"
#include "Settings.h"

#include "ViewerZoomable.h"
#include "Viewer.h"

#include "RipplingProvider.h"
#include "MandelbrotProvider.h"
#include "JuliaProvider.h"
#include "HeatTransfertProvider.h"

using std::cout;
using std::endl;

int mainGL(Settings& settings);

int mainGL(Settings& settings) {
  cout << "\n[OpenGL] mode" << endl;

  GLUTImageViewers::init(settings.getArgc(), settings.getArgv());

  Viewer<RipplingProvider> rippling(0, 0);
  ViewerZoomable<MandelbrotProvider> mandelbrot(640, 0);
  ViewerZoomable<JuliaProvider> julia(0, 640);
  Viewer<HeatTransfertProvider> heatTransfert(640, 640);

  GLUTImageViewers::runALL();  // Bloquant, Tant qu'une fenetre est ouverte

  return EXIT_SUCCESS;
}
