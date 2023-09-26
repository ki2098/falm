#ifndef FALM_MVL2_H
#define FALM_MVL2_H

#include "MVL1.h"
#include "CPM.h"

namespace Falm {

static REAL L2Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper map(pdom, Gd);
    REAL r = L0Dev_DotProduct(a, b, pdom, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static REAL L2Dev_Norm2Sq(Matrix<REAL> &a, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper map(pdom, Gd);
    REAL r = L0Dev_Norm2Sq(a, pdom, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static REAL L2Dev_MaxDiag(Matrix<REAL> &a, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper map(pdom, Gd);
    REAL r = L0Dev_MaxDiag(a, pdom, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
    return r;
}

}

#endif
