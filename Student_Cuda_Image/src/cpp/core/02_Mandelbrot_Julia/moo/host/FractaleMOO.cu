#include <iostream>
#include <omp.h>

#include "FractaleMOO.h"
#include "Device.h"
#include "Mandelbrot.h"
#include "Julia.h"

__global__ void processMandelbrot(uchar4* ptrDevPixels, int w, int h, int n, const DomaineMath& domaineMath);
__global__ void processJulia(uchar4* ptrDevPixels, int w, int h, int n, float c1, float c2, const DomaineMath& domaineMath);

FractaleMOO::FractaleMOO(int w, int h, DomaineMath* domain, Fractale* algo, int nmin, int nmax) {

	this->algo = algo;
  this->nmin = nmin;
  this->nmax = nmax;
  this->w = w;
  this->h = h;
  this->n = this->nmin;
	this->step = 1;
	this->ptrDomain = domain;

	this->dg = dim3(16, 16, 1);
	this->db = dim3(32, 32, 1);

	Device::assertDim(dg, db);

	HANDLE_ERROR(cudaMalloc(&this->ptrDevDomain, sizeof(DomaineMath)));
}

FractaleMOO::~FractaleMOO() {
	HANDLE_ERROR(cudaFree(this->ptrDevDomain));
  delete this->algo;
	delete this->ptrDomain;
}

/**
 * Override
 */
void FractaleMOO::process(uchar4* ptrDevPixels, int w, int h, const DomaineMath& domaineMath) {
	HANDLE_ERROR(cudaMemcpy(this->ptrDevDomain, &domaineMath, sizeof(DomaineMath), cudaMemcpyHostToDevice));

	if (Mandelbrot* mandelbrot = dynamic_cast<Mandelbrot*>(this->algo)) {
		processMandelbrot<<<dg,db>>>(ptrDevPixels, w, h, this->n, *this->ptrDevDomain );
	} else if (Julia* julia = dynamic_cast<Julia*>(this->algo)) {
		processJulia<<<dg,db>>>(ptrDevPixels, w, h, this->n, julia->c1, julia->c2, *this->ptrDevDomain );
	} else {
		throw "Not supported algorithm";
	}
}

/**
 * Override
 */
DomaineMath* FractaleMOO::getDomaineMathInit() {
	return this->ptrDomain;
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
	if (Mandelbrot* mandelbrot = dynamic_cast<Mandelbrot*>(this->algo)) {
		return "Fractale Mandelbrot";
	} else if (Julia* julia = dynamic_cast<Julia*>(this->algo)) {
		return "Fractale Julia";
	} else {
		return "Not supported algorithm";
	}
}
