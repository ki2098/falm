#ifndef _LS_UTIL_H_
#define _LS_UTIL_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "StructuredField.cuh"
#include "StructuredMesh.cuh"
#include "Util.cuh"

namespace FALM {

struct LS_State {
    double re;
    int    it;
};

__global__ static void poisson_sor_kernel(FieldCp<double> &a, FieldCp<double> &x, FieldCp<double> &b, double omega, int color, DomCp &dom) {
    unsigned int stride = FALMUtil::get_global_size();
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < dom._inum; idx + stride) {
        unsigned int ii, ij, ik;
        FALMUtil::idx2ijk(idx, ii, ij, ik, dom._isz);
        unsigned int oi, oj, ok;
        oi = ii + dom._guide;
        oj = ij + dom._guide;
        ok = ik + dom._guide;
        unsigned int o0, o1, o2, o3, o4, o5, o6;
        o0 = FALMUtil::ijk2idx(oi  , oj  , ok  , dom._osz);
        o1 = FALMUtil::ijk2idx(oi+1, oj  , ok  , dom._osz);
        o2 = FALMUtil::ijk2idx(oi-1, oj  , ok  , dom._osz);
        o3 = FALMUtil::ijk2idx(oi  , oj+1, ok  , dom._osz);
        o4 = FALMUtil::ijk2idx(oi  , oj-1, ok  , dom._osz);
        o5 = FALMUtil::ijk2idx(oi  , oj  , ok+1, dom._osz);
        o6 = FALMUtil::ijk2idx(oi  , oj  , ok-1, dom._osz);
        double a0, a1, a2, a3, a4, a5, a6;
        a0 = a(o0, 0);
        a1 = a(o0, 1);
        a2 = a(o0, 2);
        a3 = a(o0, 3);
        a4 = a(o0, 4);
        a5 = a(o0, 5);
        a6 = a(o0, 6);
        double x0, x1, x2, x3, x4, x5, x6;
        x0 = x(o0);
        x1 = x(o1);
        x2 = x(o2);
        x3 = x(o3);
        x4 = x(o4);
        x5 = x(o5);
        x6 = x(o6);
        double c0 = (b(o0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6)) / a0;
        if (o0 % 2 != color) {
            c0 = 0;
        }
        x(o0) = x0 + omega * c0;
    }
}

__global__ static void poisson_jacobi_kernel(FieldCp<double> &a, FieldCp<double> &xn, FieldCp<double> &xp, FieldCp<double> &b, DomCp &dom) {
    unsigned int stride = FALMUtil::get_global_size();
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < dom._inum; idx + stride) {
        unsigned int ii, ij, ik;
        FALMUtil::idx2ijk(idx, ii, ij, ik, dom._isz);
        unsigned int oi, oj, ok;
        oi = ii + dom._guide;
        oj = ij + dom._guide;
        ok = ik + dom._guide;
        unsigned int o0, o1, o2, o3, o4, o5, o6;
        o0 = FALMUtil::ijk2idx(oi  , oj  , ok  , dom._osz);
        o1 = FALMUtil::ijk2idx(oi+1, oj  , ok  , dom._osz);
        o2 = FALMUtil::ijk2idx(oi-1, oj  , ok  , dom._osz);
        o3 = FALMUtil::ijk2idx(oi  , oj+1, ok  , dom._osz);
        o4 = FALMUtil::ijk2idx(oi  , oj-1, ok  , dom._osz);
        o5 = FALMUtil::ijk2idx(oi  , oj  , ok+1, dom._osz);
        o6 = FALMUtil::ijk2idx(oi  , oj  , ok-1, dom._osz);
        double a0, a1, a2, a3, a4, a5, a6;
        a0 = a(o0, 0);
        a1 = a(o0, 1);
        a2 = a(o0, 2);
        a3 = a(o0, 3);
        a4 = a(o0, 4);
        a5 = a(o0, 5);
        a6 = a(o0, 6);
        double x0, x1, x2, x3, x4, x5, x6;
        x0 = xp(o0);
        x1 = xp(o1);
        x2 = xp(o2);
        x3 = xp(o3);
        x4 = xp(o4);
        x5 = xp(o5);
        x6 = xp(o6);
        double c0 = (b(o0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6)) / a0;
        xn(o0) = x0 + c0;
    }
}

__global__ static void calc_res_kernel(FieldCp<double> &a, FieldCp<double> &x, FieldCp<double> &b, FieldCp<double> &r, DomCp &dom) {
    unsigned int stride = FALMUtil::get_global_size();
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < dom._inum; idx + stride) {
        unsigned int ii, ij, ik;
        FALMUtil::idx2ijk(idx, ii, ij, ik, dom._isz);
        unsigned int oi, oj, ok;
        oi = ii + dom._guide;
        oj = ij + dom._guide;
        ok = ik + dom._guide;
        unsigned int o0, o1, o2, o3, o4, o5, o6;
        o0 = FALMUtil::ijk2idx(oi  , oj  , ok  , dom._osz);
        o1 = FALMUtil::ijk2idx(oi+1, oj  , ok  , dom._osz);
        o2 = FALMUtil::ijk2idx(oi-1, oj  , ok  , dom._osz);
        o3 = FALMUtil::ijk2idx(oi  , oj+1, ok  , dom._osz);
        o4 = FALMUtil::ijk2idx(oi  , oj-1, ok  , dom._osz);
        o5 = FALMUtil::ijk2idx(oi  , oj  , ok+1, dom._osz);
        o6 = FALMUtil::ijk2idx(oi  , oj  , ok-1, dom._osz);
        double a0, a1, a2, a3, a4, a5, a6;
        a0 = a(o0, 0);
        a1 = a(o0, 1);
        a2 = a(o0, 2);
        a3 = a(o0, 3);
        a4 = a(o0, 4);
        a5 = a(o0, 5);
        a6 = a(o0, 6);
        double x0, x1, x2, x3, x4, x5, x6;
        x0 = x(o0);
        x1 = x(o1);
        x2 = x(o2);
        x3 = x(o3);
        x4 = x(o4);
        x5 = x(o5);
        x6 = x(o6);
        r(o0) = b(o0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6);
    }
}

}

#endif