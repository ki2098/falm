#ifndef FALM_VECTOR_TYPES_H
#define FALM_VECTOR_TYPES_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace Falm2 {

typedef int INT;

template<typename T, size_t N>
class VECTOR {
public:
    T _mv[N];

    T &operator[](size_t _i) {return _mv[i];}

    const T &operator[](size_t _i) const {return _mv[i];}
    
    VECTOR &operator=(const VECTOR &_v) {
        for (size_t i = 0; i < N; i ++) _mv[i] = _v[i];
    }

    
};

}

#endif