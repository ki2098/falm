#ifndef FALM_FALMATH_H
#define FALM_FALMATH_H

#include <math.h>
#include "devdefine.h"

namespace Falm {

static const REAL Pi = M_PI;

__host__ __device__ static inline REAL floormod(REAL a, REAL b) {
    return a - floor(a / b) * b;
}

__host__ __device__ static inline REAL truncmod(REAL a, REAL b) {
    return a - trunc(a / b) * b;
}

}

#endif
