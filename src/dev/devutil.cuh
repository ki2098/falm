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

__device__ static inline size_t GLOBAL_THREAD_IDX() {
    size_t tnum_in_block = blockDim.x * blockDim.y * blockDim.z;
    size_t bid_in_grid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    size_t tid_in_block = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.z;
    return tnum_in_block*bid_in_grid + tid_in_block;
}

}

#endif
