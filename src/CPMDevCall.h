#ifndef FALM_CPML1V2_H
#define FALM_CPML1V2_H

#include "CPMBuffer.h"

#define CPMERR_NULLPTR -1

namespace Falm {

class CPMDevCall {
protected:
    // void CPML1Dev_PackBuffer(CPMBuffer &buffer, REAL *src, Region &pdm, dim3 block_dim);
    
    // void CPML1Dev_PackColoredBuffer(CPMBuffer &buffer, REAL *src, Region &pdm, dim3 block_dim);
    
    // void CPML1Dev_UnpackBuffer(CPMBuffer &buffer, REAL *dst, Region &pdm, dim3 block_dim);
    
    // void CPML1Dev_UnpackColoredBuffer(CPMBuffer &buffer, REAL *dst, Region &pdm, dim3 block_dim);
    
    void PackBuffer(REAL *buffer, Region &map, REAL *src, Region &pdm, dim3 block_dim, STREAM stream = (STREAM)0);
    
    void PackColoredBuffer(REAL *buffer, Region &map, INT color, REAL *src, Region &pdm, dim3 block_dim, STREAM stream = (STREAM)0);
    
    void UnpackBuffer(REAL *buffer, Region &map, REAL *dst, Region &pdm, dim3 block_dim, STREAM stream = (STREAM)0);
    
    void UnpackColoredBuffer(REAL *buffer, Region &map, INT color, REAL *dst, Region &pdm, dim3 block_dim, STREAM stream = (STREAM)0);

};

}

#endif
