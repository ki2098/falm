#ifndef FALM_TYPEDEF_H
#define FALM_TYPEDEF_H

// #define Gd   2
// #define Gdx2 4

#include <cuda.h>
#include <cuda_runtime.h>
#include "vectypes.h"
#include "nlohmann/json.hpp"

namespace Falm {

typedef double         Real;
typedef unsigned       Flag;
typedef int             Int;


typedef VECTOR<Int , 3>   Int3;
typedef VECTOR<Real, 3>  Real3;
typedef VECTOR<Int , 2>   Int2;
typedef VECTOR<Real, 2>  Real2;
typedef VECTOR<Real, 6>  Real6;

typedef cudaStream_t Stream;

typedef nlohmann::json Json;
typedef nlohmann::ordered_json OrderedJson;

// typedef int           INT_T;
// typedef char         CHAR_T;
// typedef size_t       SIZE_T;
// typedef unsigned int UINT_T;

struct FalmSnapshotInfo {
    size_t step;
    double time;
    bool tavg;
};

const Int GuideCell = 2;

}

#endif
