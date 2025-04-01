#ifndef FALM_UTIL_H
#define FALM_UTIL_H

#include "devdefine.h"

namespace Falm {

__host__ __device__ static inline Int PRODUCT3(const Int3 &u) {
    return u[0] * u[1] * u[2];
}

__host__ __device__ static inline Real PRODUCT3(const Real3 &u) {
    return u[0] * u[1] * u[2];
}

__host__ __device__ static inline Int PRODUCT3(const dim3 &u) {
    return u.x * u.y * u.z;
}

__host__ __device__ static inline Int SUM3(const Int3 &u) {
    return u[0] + u[1] + u[2];
}

__host__ __device__ static inline Int SUM3(const dim3 &u) {
    return u.x + u.y + u.z;
}


__host__ __device__ static inline Int IDX(Int i, Int j, Int k, const Int3 &shape) {
    return i + j * shape[0] + k * shape[0] * shape[1];
}

__host__ __device__ static inline size_t IDX(size_t i, size_t j, size_t k, const Vector<size_t, 3> &shape) {
    return i + j * shape[0] + k * shape[0] * shape[1];
}

__host__ __device__ static inline Int IDX(Int3 &idx, const Int3 &shape) {
    return idx[0] + idx[1] * shape[0] + idx[2] * shape[0] * shape[1];
}

__host__ __device__ static inline Int IDX(unsigned i, unsigned j, unsigned k, const dim3 &shape) {
    return i + j * shape.x + k * shape.x * shape.y;
}

__host__ __device__ static inline Int IDX(const uint3 &idx, const dim3 &shape) {
    return idx.x + idx.y * shape.x + idx.z * shape.x * shape.y;
}

}

#endif
