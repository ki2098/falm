#ifndef FALM_MVL2_H
#define FALM_MVL2_H

#include "MVL1.h"
#include "CPM.h"

namespace Falm {

static REAL L2Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL r = L0Dev_DotProduct(a, b, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, getMPIDtype<REAL>(), MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static REAL L2Dev_EuclideanNormSq(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL r = L0Dev_EuclideanNormSq(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, getMPIDtype<REAL>(), MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static REAL L2Dev_MaxDiag(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL r = L0Dev_MatColAbsMax(a, 0, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, getMPIDtype<REAL>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return r;
}

static REAL L2Dev_MatColMax(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL cmax = L0Dev_MatColMax(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&cmax, 1, getMPIDtype<REAL>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return cmax;
}

static REAL L2Dev_MatColMin(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL cmin = L0Dev_MatColMin(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&cmin, 1, getMPIDtype<REAL>(), MPI_MIN, MPI_COMM_WORLD);
    }
    return cmin;
}

static REAL L2Dev_MatColAbsMax(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL cmax = L0Dev_MatColAbsMax(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&cmax, 1, getMPIDtype<REAL>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return cmax;
}

static REAL L2Dev_MatColAbsMin(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL cmin = L0Dev_MatColAbsMin(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&cmin, 1, getMPIDtype<REAL>(), MPI_MIN, MPI_COMM_WORLD);
    }
    return cmin;
}

static REAL L2Dev_VecMax(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL vmax = L0Dev_VecMax(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&vmax, 1, getMPIDtype<REAL>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return vmax;
}

static REAL L2Dev_VecMin(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL vmax = L0Dev_VecMin(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&vmax, 1, getMPIDtype<REAL>(), MPI_MIN, MPI_COMM_WORLD);
    }
    return vmax;
}

}

#endif
