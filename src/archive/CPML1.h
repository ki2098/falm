#ifndef FALM_CPML1_H
#define FALM_CPML1_H

#include "CPMB.h"

namespace Falm {

void CPML1dev_PackBuffer(CPMBuffer<double> &buffer, double *src, Mapper &pdm, dim3 block_dim);

void CPML1dev_PackColoredBuffer(CPMBuffer<double> &buffer, double *src, Mapper &pdm, dim3 block_dim);

void CPML1dev_UnpackBuffer(CPMBuffer<double> &buffer, double *dst, Mapper &pdm, dim3 block_dim);

void CPML1dev_UnpackColoredBuffer(CPMBuffer<double> &buffer, double *dst, Mapper &pdm, dim3 block_dim);

}

#endif
