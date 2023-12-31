#ifndef FALM_DEVUTIL_CUH
#define FALM_DEVUTIL_CUH

#include "../typedef.h"

namespace Falm {

__device__ static inline void GLOBAL_THREAD_IDX_3D(unsigned &i, unsigned &j, unsigned &k) {
    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;
    k = threadIdx.z + blockIdx.z * blockDim.z;
}

__device__ static inline void GLOBAL_THREAD_IDX_3D(INT &i, INT &j, INT &k) {
    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;
    k = threadIdx.z + blockIdx.z * blockDim.z;
}

}

#endif
