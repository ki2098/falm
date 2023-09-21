#ifndef FALM_CPMDEV_H
#define FALM_CPMDEV_H

#include "CPMB.h"

namespace Falm {

void CPML1dev_PackBuffer(CPMBuffer<double> &buffer, double *src, Mapper &pdom, dim3 &block_dim);

void CPML1dev_PackColoredBuffer(CPMBuffer<double> &buffer, double *src, Mapper &pdom, dim3 &block_dim);

void CPML1dev_UnpackBuffer(CPMBuffer<double> &buffer, double *dst, Mapper &pdom, dim3 &block_dim);

void CPML1dev_UnpackColoredBuffer(CPMBuffer<double> &buffer, double *dst, Mapper &pdom, dim3 &block_dim);

}

#endif
