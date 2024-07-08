#ifndef FALM_TYPEDEF_H
#define FALM_TYPEDEF_H

// #define Gd   2
// #define Gdx2 4

#include <cuda.h>
#include <cuda_runtime.h>
#include "vectypes.h"
#include "nlohmann/json.hpp"

namespace Falm {

typedef double         REAL;
typedef unsigned       FLAG;
typedef int             INT;


typedef VECTOR<INT , 3>   INT3;
typedef VECTOR<REAL, 3>  REAL3;
typedef VECTOR<INT , 2>   INT2;
typedef VECTOR<REAL, 2>  REAL2;
typedef VECTOR<REAL, 6>  REAL6;

typedef cudaStream_t STREAM;

typedef nlohmann::json json;
typedef nlohmann::ordered_json ordered_json;

// typedef int           INT_T;
// typedef char         CHAR_T;
// typedef size_t       SIZE_T;
// typedef unsigned int UINT_T;

struct FalmSnapshotInfo {
    size_t step;
    double time;
    bool tavg;
};

const INT GuideCell = 2;

}

#endif
