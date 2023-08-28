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
    unsigned int N = size.x * size.y * size.z;
    if (
        i >= guide && i < size.x - guide &&
        j >= guide && j < size.y - guide &&
        k >= guide && k < size.z - guide
    ) {
        
    }
}

}

#endif