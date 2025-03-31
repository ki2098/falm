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
    
    void PackBuffer(Real *buffer, Region &map, Real *src, Region &pdm, dim3 block_dim, Stream stream = (Stream)0);
    
    void PackColoredBuffer(Real *buffer, Region &map, Int color, Real *src, Region &pdm, dim3 block_dim, Stream stream = (Stream)0);
    
    void UnpackBuffer(Real *buffer, Region &map, Real *dst, Region &pdm, dim3 block_dim, Stream stream = (Stream)0);
    
    void UnpackColoredBuffer(Real *buffer, Region &map, Int color, Real *dst, Region &pdm, dim3 block_dim, Stream stream = (Stream)0);

};

}

#endif
