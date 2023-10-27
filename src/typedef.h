#ifndef FALM_TYPEDEF_H
#define FALM_TYPEDEF_H

// #define Gd   2
// #define Gdx2 4

#include <cuda.h>
#include <cuda_runtime.h>
#include "vectypes.h"

namespace Falm {

typedef double         REAL;
typedef unsigned       FLAG;
typedef int             INT;


typedef VECTOR3<INT>   INT3;
typedef VECTOR3<REAL> REAL3;
typedef VECTOR2<INT>   INT2;
typedef VECTOR2<REAL> REAL2;

typedef cudaStream_t STREAM;

// typedef int           INT_T;
// typedef char         CHAR_T;
// typedef size_t       SIZE_T;
// typedef unsigned int UINT_T;

const INT GuideCell = 2;

}

#endif
