#ifndef FALM_CPML1V2_H
#define FALM_CPML1V2_H

#include "CPMBv2.h"

#define CPMERR_NULLPTR -1

namespace Falm {

void CPML1Dev_PackBuffer(CPMBuffer &buffer, REAL *src, Mapper &pdm, dim3 block_dim);

void CPML1Dev_PackColoredBuffer(CPMBuffer &buffer, REAL *src, Mapper &pdm, dim3 block_dim);

void CPML1Dev_UnpackBuffer(CPMBuffer &buffer, REAL *dst, Mapper &pdm, dim3 block_dim);

void CPML1Dev_UnpackColoredBuffer(CPMBuffer &buffer, REAL *dst, Mapper &pdm, dim3 block_dim);

void CPML0Dev_PackBuffer(REAL *buffer, Mapper &map, REAL *src, Mapper &pdm, dim3 block_dim, STREAM stream = (STREAM)0);

void CPML0Dev_PackColoredBuffer(REAL *buffer, Mapper &map, INT color, REAL *src, Mapper &pdm, dim3 block_dim, STREAM stream = (STREAM)0);

void CPML0Dev_UnpackBuffer(REAL *buffer, Mapper &map, REAL *dst, Mapper &pdm, dim3 block_dim, STREAM stream = (STREAM)0);

void CPML0Dev_UnpackColoredBuffer(REAL *buffer, Mapper &map, INT color, REAL *dst, Mapper &pdm, dim3 block_dim, STREAM stream = (STREAM)0);

}

#endif
