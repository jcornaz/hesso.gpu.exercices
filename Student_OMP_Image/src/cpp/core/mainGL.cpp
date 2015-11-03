#include <iostream>

#include "GLUTImageViewers.h"
#include "Settings.h"

#include "ViewerZoomable.h"
#include "Viewer.h"

#include "RayTracingProvider.h"

using std::cout;
using std::endl;

int mainGL(Settings& settings);

int mainGL(Settings& settings) {
  cout << "\n[OpenGL] mode" << endl;

  GLUTImageViewers::init(settings.getArgc(), settings.getArgv()); // call once

  // Viewer : (int,int,boolean) : (px,py,isAnimation=true)
  Viewer<RayTracingProvider> raytracing( 0, 0);
  // add other viewer here!

  GLUTImageViewers::runALL();  // Bloquant, Tant qu'une fenetre est ouverte

  return EXIT_SUCCESS;
}
