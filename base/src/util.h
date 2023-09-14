#ifndef FALM_UTIL_H
#define FALM_UTIL_H

#include "typedef.h"

namespace Falm {

__host__ __device__ static inline unsigned int PRODUCT3(uint3 &u) {
    return u.x * u.y * u.z;
}

}

#endif