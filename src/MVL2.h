#ifndef FALM_MVL2_H
#define FALM_MVL2_H

#include "MVL1.h"
#include "CPM.h"

namespace Falm {

static REAL L2Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, dim3 block_dim, CPMBase &cpm) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL r = L0Dev_DotProduct(a, b, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static REAL L2Dev_EuclideanNormSq(Matrix<REAL> &a, dim3 block_dim, CPMBase &cpm) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL r = L0Dev_EuclideanNormSq(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static REAL L2Dev_MaxDiag(Matrix<REAL> &a, dim3 block_dim, CPMBase &cpm) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL r = L0Dev_MaxDiag(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
    return r;
}

}

#endif
