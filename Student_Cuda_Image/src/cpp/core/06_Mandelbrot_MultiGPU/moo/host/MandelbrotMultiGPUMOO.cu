#include <iostream>
#include <omp.h>

#include "MandelbrotMultiGPUMOO.h"
#include "Device.h"
#include "Mandelbrot.h"

__global__ void processMandelbrot(uchar4* ptrDevPixels, int w, int h, int n, const DomaineMath& domaineMath);
__global__ void processJulia(uchar4* ptrDevPixels, int w, int h, int n, float c1, float c2, const DomaineMath& domaineMath);

MandelbrotMultiGPUMOO::MandelbrotMultiGPUMOO(int w, int h, DomaineMath* domain, Fractale* algo, int nmin, int nmax) {

	this->algo = algo;
  this->nmin = nmin;
  this->nmax = nmax;
  this->w = w;
  this->h = h;
  this->n = this->nmin;
	this->step = 1;
	this->ptrDomain = domain;

	this->nbDevices = Device::getDeviceCount();

	this->dg = (dim3*) malloc(sizeof(dim3) * this->nbDevices);
	this->db = (dim3*) malloc(sizeof(dim3) * this->nbDevices);

	this->ptrDevDomains = (DomaineMath**) malloc(sizeof(DomaineMath*) * this->nbDevices);

	#pragma omp parallel for
	for (int i = 0 ; i < this->nbDevices ; i++) {
		HANDLE_ERROR(cudaSetDevice(i));

		this->dg[i] = dim3(16, 16, 1);
		this->db[i] = dim3(32, 32, 1);

		Device::assertDim(this->dg[i], this->db[i]);

		HANDLE_ERROR(cudaMalloc(&this->ptrDevDomains[i], sizeof(DomaineMath)));
	}
}

MandelbrotMultiGPUMOO::~MandelbrotMultiGPUMOO() {

	#pragma omp parallel for
	for (int i = 0 ; i < this->nbDevices ; i++ ) {
		HANDLE_ERROR(cudaFree(this->ptrDevDomains[i]));
	}

	free(this->dg);
	free(this->db);
	free(this->ptrDevDomains);

	delete this->algo;
	delete this->ptrDomain;
}

/**
 * Override
 */
void MandelbrotMultiGPUMOO::process(uchar4* ptrDevPixels, int w, int h, const DomaineMath& domaineMath) {
	const int hPix = h / this->nbDevices;
	const float hDom = (domaineMath.y1 - domaineMath.y0) / this->nbDevices;
	const int tabSize = hPix * w;

	#pragma omp parallel for
	for (int i = 0 ; i < this->nbDevices ; i++) {
		HANDLE_ERROR(cudaSetDevice(i));

		DomaineMath deviceDomain;

		deviceDomain.x0 = domaineMath.x0;
		deviceDomain.x1 = domaineMath.x1;
		deviceDomain.y0 = domaineMath.y0 + i * hDom;
		deviceDomain.y1 = deviceDomain.y0 + hDom;

		uchar4* ptrDevPixelsLocal;

		HANDLE_ERROR(cudaMalloc(&ptrDevPixelsLocal, sizeof(uchar4) * tabSize));
		HANDLE_ERROR(cudaMemcpy(ptrDevPixelsLocal, &ptrDevPixels[i * tabSize], sizeof(uchar4) * tabSize, cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMemcpy(this->ptrDevDomains[i], &deviceDomain, sizeof(DomaineMath), cudaMemcpyHostToDevice));

		processMandelbrot<<<this->dg[i],this->db[i]>>>(ptrDevPixelsLocal, w, hPix, this->n, *this->ptrDevDomains[i] );

		HANDLE_ERROR(cudaMemcpy(&ptrDevPixels[i * tabSize], ptrDevPixelsLocal, sizeof(uchar4) * tabSize, cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaFree(ptrDevPixelsLocal));
	}
}

/**
 * Override
 */
DomaineMath* MandelbrotMultiGPUMOO::getDomaineMathInit() {
	return this->ptrDomain;
}

/**
 * Override
 */
void MandelbrotMultiGPUMOO::animationStep() {

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
float MandelbrotMultiGPUMOO::getAnimationPara() {
	return (float) this->n;
}

/**
 * Override
 */
int MandelbrotMultiGPUMOO::getW()	{
	return this->w;
}

/**
 * Override
 */
int MandelbrotMultiGPUMOO::getH() {
	return this->h;
}

/**
 * Override
 */
string MandelbrotMultiGPUMOO::getTitle() {
	return "Fractale Mandelbrot multi GPU";
}
