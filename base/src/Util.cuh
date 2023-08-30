#ifndef FALM_UTIL_CUH
#define FALM_UTIL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

namespace FALM {

struct MPI_State {
    int size;
    int rank;
};

namespace SYNC {

static const unsigned int H2D = 0;
static const unsigned int D2H = 1;

}

namespace LOC {

static const unsigned int NONE   = 0;
static const unsigned int HOST   = 1;
static const unsigned int DEVICE = 2;
static const unsigned int BOTH   = HOST | DEVICE;

}

namespace UTIL {

__host__ __device__ static inline unsigned int IDX(unsigned int i, unsigned int j, unsigned int k, dim3 &size) {
    return i + j * size.x + k * size.x * size.y;
}

__host__ __device__ static inline unsigned int IDX(unsigned int i, unsigned int j, unsigned int k, unsigned int m, dim3 &size, unsigned int dim) {
    return i + j * size.x + k * size.x * size.y + m * size.x * size.y * size.z;
}

__device__ static inline void THREAD2IJK(unsigned int &i, unsigned int &j, unsigned int &k) {
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;
}

unsigned int flip_color(unsigned int color) {
    return color ^ 1U;
}

}

}

#endif