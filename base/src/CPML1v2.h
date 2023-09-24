#ifndef FALM_CPML1V2_H
#define FALM_CPML1V2_H

#include "CPMBv2.h"

namespace Falm {

void CPML1dev_PackBuffer(CPMBuffer &buffer, double *src, Mapper &pdom, dim3 block_dim);

void CPML1dev_PackColoredBuffer(CPMBuffer &buffer, double *src, Mapper &pdom, dim3 block_dim);

void CPML1dev_UnpackBuffer(CPMBuffer &buffer, double *dst, Mapper &pdom, dim3 block_dim);

void CPML1dev_UnpackColoredBuffer(CPMBuffer &buffer, double *dst, Mapper &pdom, dim3 block_dim);

}

#endif