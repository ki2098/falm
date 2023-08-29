#ifndef FALM_POI_CUH
#define FALM_POI_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include "param.h"
#include "Util.cuh"
#include "Field.cuh"
#include "Mapper.cuh"
#include "CPM.cuh"

namespace FALM {

struct LS_State {
    double re;
    int    it;
};

__global__ static void poisson_sor_kernel(FieldFrame<double> &a, FieldFrame<double> &x, FieldFrame<double> &b, int color, Mapper domain, Mapper range) {
    unsigned int i, j, k;
    UTIL::THREAD2IJK(i, j, k);
    dim3 &size = domain.size;
    dim3 &origin = domain.offset;
    if (i < range.size.x && j < range.size.y && k < range.size.z) {
        i += range.offset.x;
        j += range.offset.y;
        k += range.offset.z;
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

__global__ void poisson_jacobi_kernel(FieldFrame<double> &a, FieldFrame<double> &xn, FieldFrame<double> &xp, FieldFrame<double> &b, Mapper domain, Mapper range) {
    unsigned int i, j, k;
    UTIL::THREAD2IJK(i, j, k);
    dim3 &size = domain.size;
    dim3 &origin = domain.offset;
    if (i < range.size.x && j < range.size.y && k < range.size.z) {
        i += range.offset.x;
        j += range.offset.y;
        k += range.offset.z;
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
        x0 = xp(id0);
        x1 = xp(id1);
        x2 = xp(id2);
        x3 = xp(id3);
        x4 = xp(id4);
        x5 = xp(id5);
        x6 = xp(id6);
        double c0 = (b(id0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6)) / a0;
        xn(id0) = x0 + c0;
    }
}

__global__ static void res_kernel(FieldFrame<double> &a, FieldFrame<double> &x, FieldFrame<double> &b, FieldFrame<double> &r, Mapper domain, Mapper range) {
    unsigned int i, j, k;
    UTIL::THREAD2IJK(i, j, k);
    dim3 &size = domain.size;
    dim3 &origin = domain.offset;
    if (i < range.size.x && j < range.size.y && k < range.size.z) {
        i += range.offset.x;
        j += range.offset.y;
        k += range.offset.z;
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
        r(id0) = b(id0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6);
    }
}


}

#endif