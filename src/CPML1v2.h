#ifndef FALM_CPML1V2_H
#define FALM_CPML1V2_H

#include "CPMBv2.h"

#define CPMERR_NULLPTR -1

namespace Falm {

void CPML1Dev_PackBuffer(CPMBuffer &buffer, REAL *src, Region &pdm, dim3 block_dim);

void CPML1Dev_PackColoredBuffer(CPMBuffer &buffer, REAL *src, Region &pdm, dim3 block_dim);

void CPML1Dev_UnpackBuffer(CPMBuffer &buffer, REAL *dst, Region &pdm, dim3 block_dim);

void CPML1Dev_UnpackColoredBuffer(CPMBuffer &buffer, REAL *dst, Region &pdm, dim3 block_dim);

void CPML0Dev_PackBuffer(REAL *buffer, Region &map, REAL *src, Region &pdm, dim3 block_dim, STREAM stream = (STREAM)0);

void CPML0Dev_PackColoredBuffer(REAL *buffer, Region &map, INT color, REAL *src, Region &pdm, dim3 block_dim, STREAM stream = (STREAM)0);

void CPML0Dev_UnpackBuffer(REAL *buffer, Region &map, REAL *dst, Region &pdm, dim3 block_dim, STREAM stream = (STREAM)0);

void CPML0Dev_UnpackColoredBuffer(REAL *buffer, Region &map, INT color, REAL *dst, Region &pdm, dim3 block_dim, STREAM stream = (STREAM)0);

}

#endif
