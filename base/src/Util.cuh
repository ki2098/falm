#ifndef _UTIL_H_
#define _UTIL_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>

namespace FALMUtil {

__host__ __device__ static inline void d123(unsigned int idx, unsigned int &i, unsigned int &j, unsigned int &k, dim3 &size) {
    unsigned int slice = size.y * size.z;
    i   = idx / slice;
    idx = idx % slice;
    j   = idx / size.z;
    k   = idx % size.z;
}

__host__ __device__ static inline unsigned int d321(unsigned int i, unsigned int j, unsigned int k, dim3 &size) {
    return i * size.y * size.z + j * size.z + k;
}

__device__ static inline unsigned int get_global_idx() {
    return blockDim.x * blockIdx.x + threadIdx.x;
}

__device__ static inline unsigned int get_global_size() {
    return blockDim.x * gridDim.x;
}

}

#endif