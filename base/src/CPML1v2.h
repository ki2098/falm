#ifndef FALM_CPML1V2_H
#define FALM_CPML1V2_H

#include "CPMBv2.h"

#define CPMERR_NULLPTR -1

namespace Falm {

void CPML1Dev_PackBuffer(CPMBuffer &buffer, double *src, Mapper &pdom, dim3 block_dim);

void CPML1Dev_PackColoredBuffer(CPMBuffer &buffer, double *src, Mapper &pdom, dim3 block_dim);

void CPML1Dev_UnpackBuffer(CPMBuffer &buffer, double *dst, Mapper &pdom, dim3 block_dim);

void CPML1Dev_UnpackColoredBuffer(CPMBuffer &buffer, double *dst, Mapper &pdom, dim3 block_dim);

}

#endif
