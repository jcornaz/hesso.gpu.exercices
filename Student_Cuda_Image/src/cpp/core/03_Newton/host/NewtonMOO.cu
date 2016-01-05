#include <iostream>
#include <omp.h>

#include "NewtonMOO.h"
#include "OmpTools.h"
#include "IndiceTools.h"
#include "Device.h"

extern __global__ void newton(uchar4* ptrDevTabPixels, int w, int h, int n, DomaineMath* ptrDevDomain, NewtonMath* ptrDevMath);

NewtonMOO::NewtonMOO(int w, int h, DomaineMath* ptrDomain, NewtonMath* ptrMath) {
	this->ptrDomain = ptrDomain;

  this->w = w;
  this->h = h;
	this->n = 0;

	this->dg = dim3(64, 64, 1);
	this->db = dim3(32, 32, 1);

	Device::assertDim(dg, db);

	HANDLE_ERROR(cudaMalloc(&this->ptrDevDomain, sizeof(DomaineMath)));
	HANDLE_ERROR(cudaMalloc(&this->ptrDevMath, sizeof(NewtonMath)));
	HANDLE_ERROR(cudaMemcpy(this->ptrDevMath, ptrMath, sizeof(NewtonMath), cudaMemcpyHostToDevice));

	delete ptrMath;
}

NewtonMOO::~NewtonMOO() {
	delete this->ptrDomain;
	HANDLE_ERROR(cudaFree(this->ptrDevDomain));
	HANDLE_ERROR(cudaFree(this->ptrDevMath));
}

/**
 * Override
 */
void NewtonMOO::process( uchar4* ptrDevTabPixels, int w, int h, const DomaineMath& domaineMath ) {
	HANDLE_ERROR(cudaMemcpy(this->ptrDevDomain, &domaineMath, sizeof(DomaineMath), cudaMemcpyHostToDevice));

	newton<<<dg,db>>>(ptrDevTabPixels, w, h, this->n, this->ptrDevDomain, this->ptrDevMath);
}

/**
 * Override
 */
DomaineMath* NewtonMOO::getDomaineMathInit() {
	return this->ptrDomain;
}

/**
 * Override
 */
void NewtonMOO::animationStep() {
	this->n++;
}

/**
 * Override
 */
float NewtonMOO::getAnimationPara() {
	return (float) this->n;
}

/**
 * Override
 */
int NewtonMOO::getW()	{
	return this->w;
}

/**
 * Override
 */
int NewtonMOO::getH() {
	return this->h;
}

/**
 * Override
 */
string NewtonMOO::getTitle() {
	return "Fractale Newton";
}
