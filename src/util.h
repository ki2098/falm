#ifndef FALM_UTIL_H
#define FALM_UTIL_H

#include "devdefine.h"

namespace Falm {

__host__ __device__ static inline INT PRODUCT3(const INT3 &u) {
    return u[0] * u[1] * u[2];
}

__host__ __device__ static inline REAL PRODUCT3(const REAL3 &u) {
    return u[0] * u[1] * u[2];
}

__host__ __device__ static inline INT PRODUCT3(const dim3 &u) {
    return u.x * u.y * u.z;
}

__host__ __device__ static inline INT SUM3(const INT3 &u) {
    return u[0] + u[1] + u[2];
}

__host__ __device__ static inline INT SUM3(const dim3 &u) {
    return u.x + u.y + u.z;
}


__host__ __device__ static inline INT IDX(INT i, INT j, INT k, const INT3 &shape) {
    return i + j * shape[0] + k * shape[0] * shape[1];
}

__host__ __device__ static inline size_t IDX(size_t i, size_t j, size_t k, const VECTOR<size_t, 3> &shape) {
    return i + j * shape[0] + k * shape[0] * shape[1];
}

__host__ __device__ static inline INT IDX(INT3 &idx, const INT3 &shape) {
    return idx[0] + idx[1] * shape[0] + idx[2] * shape[0] * shape[1];
}

__host__ __device__ static inline INT IDX(unsigned i, unsigned j, unsigned k, const dim3 &shape) {
    return i + j * shape.x + k * shape.x * shape.y;
}

__host__ __device__ static inline INT IDX(const uint3 &idx, const dim3 &shape) {
    return idx.x + idx.y * shape.x + idx.z * shape.x * shape.y;
}

}

#endif
