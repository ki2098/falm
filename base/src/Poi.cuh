#ifndef FALM_POI_CUH
#define FALM_POI_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include "param.h"
#include "Util.cuh"
#include "Field.cuh"

namespace FALM {

struct LS_State {
    double re;
    int    it;
};

__global__ static void poisson_sor_kernel(FieldFrame<double> &a, FieldFrame<double> &x, FieldFrame<double> &b, int color, dim3 size, dim3 origin) {
    unsigned int i, j, k;
    UTIL::get_global_idx(i, j, k);
    unsigned int N = size.x * size.y * size.z;
    if (
        i >= guide && i < size.x - guide &&
        j >= guide && j < size.y - guide &&
        k >= guide && k < size.z - guide
    ) {
        unsigned int id0, id1, id2, id3, id4, id5, id6;
        id0 = UTIL::IDX(i  , j  , k  , size);
        id1 = UTIL::IDX(i+1, j  , k  , size);
        id2 = UTIL::IDX(i-1, j  , k  , size);
        id3 = UTIL::IDX(i  , j+1, k  , size);
        id4 = UTIL::IDX(i  , j-1, k  , size);
        id5 = UTIL::IDX(i  , j  , k+1, size);
        id6 = UTIL::IDX(i  , j  , k-1, size);
        double a0, a1, a2, a3, a4, a5, a6;
        a0 = a(id0, 0);
        a1 = a(id0, 1);
        a2 = a(id0, 2);
        a3 = a(id0, 3);
        a4 = a(id0, 4);
        a5 = a(id0, 5);
        a6 = a(id0, 6);
        double x0, x1, x2, x3, x4, x5, x6;
        x0 = x(id0);
        x1 = x(id1);
        x2 = x(id2);
        x3 = x(id3);
        x4 = x(id4);
        x5 = x(id5);
        x6 = x(id6);
        double c0 = (b(id0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6)) / a0;
        if ((i + j + k + origin.x + origin.y + origin.z) % 2 != color) {
            c0 = 0;
        }
        x(id0) = x0 + sor_omega * c0;
    }
}

}

#endif