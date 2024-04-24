#ifndef FALM_FALMATH_H
#define FALM_FALMATH_H

#include <math.h>
#include "devdefine.h"

namespace Falm {

static const REAL Pi = M_PI;

__host__ __device__ static inline REAL square(REAL a) {
    return a * a;
}

__host__ __device__ static inline REAL cube(REAL a) {
    return a * a * a;
}

__host__ __device__ static inline REAL floormod(REAL a, REAL b) {
    return a - floor(a / b) * b;
}

__host__ __device__ static inline REAL truncmod(REAL a, REAL b) {
    return a - trunc(a / b) * b;
}

template<typename T>
__host__ __device__ static inline INT sign(T a) {
    return (a>0) - (a<0);
}

}

#endif
