#ifndef FALM_CPML1V2_H
#define FALM_CPML1V2_H

#include "CPMBv2.h"

#define CPMERR_NULLPTR -1

namespace Falm {

void CPML1Dev_PackBuffer(CPMBuffer &buffer, REAL *src, Mapper &pdom, dim3 block_dim);

void CPML1Dev_PackColoredBuffer(CPMBuffer &buffer, REAL *src, Mapper &pdom, dim3 block_dim);

void CPML1Dev_UnpackBuffer(CPMBuffer &buffer, REAL *dst, Mapper &pdom, dim3 block_dim);

void CPML1Dev_UnpackColoredBuffer(CPMBuffer &buffer, REAL *dst, Mapper &pdom, dim3 block_dim);

}

#endif
