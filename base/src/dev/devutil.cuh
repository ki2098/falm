#ifndef FALM_DEVUTIL_CUH
#define FALM_DEVUTIL_CUH

#include <cuda_runtime.h>

namespace Falm {

__device__ static inline void GLOBAL_THREAD_IDX_3D(unsigned int &i, unsigned int &j, unsigned int &k) {
    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;
    k = threadIdx.z + blockIdx.z * blockDim.z;
}

}

#endif