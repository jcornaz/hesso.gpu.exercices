#include <iostream>

#include "GLUTImageViewers.h"
#include "Settings.h"

#include "ViewerZoomable.h"
#include "Viewer.h"

#include "ConvolutionProvider.h"

using std::cout;
using std::endl;

int mainGL(Settings& settings);

int mainGL(Settings& settings) {
  cout << "\n[OpenGL] mode" << endl;

  GLUTImageViewers::init(settings.getArgc(), settings.getArgv());

  Viewer<ConvolutionProvider> convolution(0, 0);

  GLUTImageViewers::runALL();  // Bloquant, Tant qu'une fenetre est ouverte

  return EXIT_SUCCESS;
}
