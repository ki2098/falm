#ifndef FALM_TYPEDEF_H
#define FALM_TYPEDEF_H

#define Gd   2
#define Gdx2 4

#include <cuda.h>
#include <cuda_runtime.h>

namespace Falm {

typedef double         REAL;
typedef uint32_t       FLAG;
typedef int32_t         INT;
struct INTx3 {INT x, y, z;};
struct __attribute__((aligned(8))) INTx2 {INT x, y;};

// typedef int           INT_T;
// typedef char         CHAR_T;
// typedef size_t       SIZE_T;
// typedef unsigned int UINT_T;

}

#endif
