#include <iostream>
#include <omp.h>

#include "FractaleMOO.h"
#include "OmpTools.h"
#include "IndiceTools.h"
#include "Device.h"
#include "Mandelbrot.h"
#include "Julia.h"

using std::cout;
using std::endl;

__global__ void processMandelbrot(uchar4* ptrDevPixels, int w, int h, int n, const DomaineMath& domaineMath);
__global__ void processJulia(uchar4* ptrDevPixels, int w, int h, int n, float c1, float c2, const DomaineMath& domaineMath);

FractaleMOO::FractaleMOO(int w, int h, DomaineMath* domain, Fractale* algo, int nmin, int nmax) {
	this->algo = algo;
	this->domain = domain;
  this->nmin = nmin;
  this->nmax = nmax;
  this->w = w;
  this->h = h;
  this->n = this->nmin;
	this->step = 1;

	this->dg = dim3(8, 8, 1);
	this->db = dim3(16, 16, 1);

	Device::assertDim(dg, db);
}

FractaleMOO::~FractaleMOO() {
  delete this->algo;
	delete this->domain;
}

/**
 * Override
 */
void FractaleMOO::process(uchar4* ptrDevPixels, int w, int h, const DomaineMath& domaineMath) {

		// if (Mandelbrot* mandelbrot = dynamic_cast<Mandelbrot*>(this->algo)) {
			processMandelbrot<<<dg,db>>>(ptrDevPixels, w, h, this->n, domaineMath );
		// } else if (Julia* julia = dynamic_cast<Julia*>(this->algo)) {
		// 	processJulia<<<dg,db>>>(ptrDevPixels, w, h, this->n, julia->c1, julia->c2, domaineMath );
		// } else {
		// 	throw "Not supported algorithm";
		// }
	// cout << cudaGetErrorString( cudaGetLastError() ) << endl;
}

DomaineMath* FractaleMOO::getDomaineMathInit() {
	return this->domain;
}

/**
 * Override
 */
void FractaleMOO::animationStep() {

	if( this->n == this->nmax ) {
		this->step = -1;
	} else if(this->n == this->nmin ) {
		this->step = 1;
	}

	this->n += this->step;
}

/**
 * Override
 */
float FractaleMOO::getAnimationPara() {
	return (float) this->n;
}

/**
 * Override
 */
int FractaleMOO::getW()	{
	return this->w;
}

/**
 * Override
 */
int FractaleMOO::getH() {
	return this->h;
}

/**
 * Override
 */
string FractaleMOO::getTitle() {
	return "Fractale Mandelbrot";
}
