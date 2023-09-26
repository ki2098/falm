#ifndef FALM_UTIL_H
#define FALM_UTIL_H

#include "typedef.h"

namespace Falm {

__host__ __device__ static inline unsigned int PRODUCT3(const uint3 &u) {
    return u.x * u.y * u.z;
}

__host__ __device__ static inline INT PRODUCT3(const INTx3 &u) {
    return u.x * u.y * u.z;
}

__host__ __device__ static inline unsigned int PRODUCT3(const dim3 &u) {
    return u.x * u.y * u.z;
}

__host__ __device__ static inline unsigned int SUM3(const uint3 &u) {
    return u.x + u.y + u.z;
}

__host__ __device__ static inline INT SUM3(const INTx3 &u) {
    return u.x + u.y + u.z;
}

__host__ __device__ static inline unsigned int SUM3(const dim3 &u) {
    return u.x + u.y + u.z;
}

__host__ __device__ static inline unsigned int IDX(unsigned int i, unsigned int j, unsigned int k, const uint3 &shape) {
    return i + j * shape.x + k * shape.x * shape.y;
}

__host__ __device__ static inline INT IDX(INT i, INT j, INT k, const INTx3 &shape) {
    return i + j * shape.x + k * shape.x * shape.y;
}

__host__ __device__ static inline unsigned int IDX(const uint3 &idx, const uint3 &shape) {
    return idx.x + idx.y * shape.x + idx.z * shape.x * shape.y;
}

__host__ __device__ static inline INT IDX(INTx3 &idx, const INTx3 &shape) {
    return idx.x + idx.y * shape.x + idx.z * shape.x * shape.y;
}

__host__ __device__ static inline unsigned int IDX(unsigned int i, unsigned int j, unsigned int k, const dim3 &shape) {
    return i + j * shape.x + k * shape.x * shape.y;
}

__host__ __device__ static inline unsigned int IDX(const uint3 &idx, const dim3 &shape) {
    return idx.x + idx.y * shape.x + idx.z * shape.x * shape.y;
}

}

#endif
