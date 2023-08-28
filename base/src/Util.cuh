#ifndef FALM_UTIL_CUH
#define FALM_UTIL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

namespace FALM {

namespace LOC {

static const unsigned int NONE   = 0;
static const unsigned int HOST   = 1;
static const unsigned int DEVICE = 2;
static const unsigned int BOTH   = HOST | DEVICE;

}

namespace UTIL {

__host__ __device__ static inline unsigned int IDX(unsigned int i, unsigned int j, unsigned int k, dim3 &size) {
    return i * size.y * size.z + j * size.z + k;
}

__host__ __device__ static inline unsigned int IDX(unsigned int i, unsigned int j, unsigned int k, unsigned int m, dim3 &size, unsigned int dim) {
    return m * size.x * size.y * size.z + i * size.y * size.z + j * size.z + k;
}

__host__ __device__ static inline unsigned int IDX(unsigned int i, unsigned int m, unsigned int len, unsigned int dim) {
    return m * len + i;
}

__device__ static inline void get_global_idx(unsigned int &i, unsigned int &j, unsigned int &k) {
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;
}

}

}

#endif