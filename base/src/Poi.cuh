#ifndef FALM_POI_CUH
#define FALM_POI_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include "param.h"
#include "Util.cuh"

namespace FALM {

struct LS_State {
    double re;
    int    it;
};

__global__ static void poisson_sor_kernel(double *a, double *x, double *b, double omega, int color, dim3 size, dim3 origin) {
    unsigned int i, j, k;
    UTIL::get_global_idx(i, j, k);
    unsigned int len = size.x * size.y * size.z;
    if (
        i >= guide && i < size.x - guide &&
        j >= guide && j < size.y - guide &&
        k >= guide && k < size.z - guide
    ) {
        unsigned int o0, o1, o2, o3, o4, o5, o6;
        o0 = UTIL::IDX(i  , j  , k  , size);
        o1 = UTIL::IDX(i+1, j  , k  , size);
        o2 = UTIL::IDX(i-1, j  , k  , size);
        o3 = UTIL::IDX(i  , j+1, k  , size);
        o4 = UTIL::IDX(i  , j-1, k  , size);
        o5 = UTIL::IDX(i  , j  , k+1, size);
        o6 = UTIL::IDX(i  , j  , k-1, size);
        
    }
}

}

#endif