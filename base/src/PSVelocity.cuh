#ifndef _PSVELOCITY_H_
#define _PSVELOCITY_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <mpi.h>
#include "StructuredField.cuh"
#include "Util.cuh"

namespace FALM {

__device__ static double upwind3(double u00, double u11, double u12, double u13, double u14, double u21, double u22, double u23, double u24, double u31, double u32, double u33, double u34, double abs1, double abs2, double abs3, double uu11, double uu12, double uu21, double uu22, double uu31, double uu32, double j00) {
    double adv = 0;
    double j2  = 2 * j00;
    adv += uu12 * (- u14 + 27 * u13 - 27 * u00 + u12) / j2;
    adv += uu11 * (- u13 + 27 * u00 - 27 * u12 + u11) / j2;
    adv += abs1 * (u14 - 4 * u13 + 6 * u00 - 4 * u12 + u11);
    adv += uu22 * (- u24 + 27 * u23 - 27 * u00 + u22) / j2;
    adv += uu21 * (- u23 + 27 * u00 - 27 * u22 + u21) / j2;
    adv += abs2 * (u24 - 4 * u23 + 6 * u00 - 4 * u22 + u21);
    adv += uu32 * (- u34 + 27 * u33 - 27 * u00 + u32) / j2;
    adv += uu31 * (- u33 + 27 * u00 - 27 * u32 + u31) / j2;
    adv += abs3 * (u34 - 4 * u33 + 6 * u00 - 4 * u32 + u31);
    adv /= 24.0;
    return adv;
}

__device__ static double diffusion(double u00, double u11, double u12, double u21, double u22, double u31, double u32, double n00, double n11, double n12, double n21, double n22, double n31, double n32, double g10, double g11, double g12,double g20, double g21, double g22, double g30, double g31, double g32, double j00) {
    double vis = 0;
    vis += (g12 + g10) * (ri + 0.5 * (n00 + n12)) * (u12 - u00);
    vis -= (g11 + g10) * (ri + 0.5 * (n00 + n11)) * (u00 - u11);
    vis += (g22 + g20) * (ri + 0.5 * (n00 + n22)) * (u22 - u00);
    vis -= (g21 + g20) * (ri + 0.5 * (n00 + n21)) * (u00 - u21);
    vis += (g32 + g30) * (ri + 0.5 * (n00 + n32)) * (u32 - u00);
    vis -= (g31 + g30) * (ri + 0.5 * (n00 + n31)) * (u00 - u31);
    vis /= (2 * j00);
    return vis;
}

__global__ static void psvelocity_kernel(FieldCp<double> &u, FieldCp<double> &uu, FieldCp<double> &ua, FieldCp<double> &nut, FieldCp<double> &kx, FieldCp<double> &g, FieldCp<double> &ja, FieldCp<double> &ff, DomCp &dom, unsigned int idx_start, unsigned int idx_end) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &isz = dom._isz;
    dim3 &osz = dom._osz;
    unsigned int guide = dom._guide;
    for (unsigned int idx = FALMUtil::get_global_idx() + idx_start; idx < idx_end; idx += stride) {
        unsigned int ii, ij, ik;
        FALMUtil::d123(idx, ii, ij, ik, isz);
        unsigned int i, j, k;
        i = ii + guide;
        j = ij + guide;
        k = ik + guide;
        
    }
}

}

#endif