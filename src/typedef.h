#ifndef FALM_TYPEDEF_H
#define FALM_TYPEDEF_H

#define Gd   2
#define Gdx2 4

#include <cuda.h>
#include <cuda_runtime.h>

namespace Falm {

typedef double         REAL;
typedef unsigned       FLAG;
typedef int             INT;
typedef int3          INTx3;
typedef int2          INTx2;

typedef cudaStream_t STREAM;

// typedef int           INT_T;
// typedef char         CHAR_T;
// typedef size_t       SIZE_T;
// typedef unsigned int UINT_T;

}

#endif
