#ifndef FALM_MVL2_H
#define FALM_MVL2_H

#include "MVL1.h"
#include "CPML2v2.h"

namespace Falm {

static double devL2_DotProduct(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper map(pdom, Gd);
    double r = devL0_DotProduct(a, b, pdom, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static double devL2_Norm2Sq(Matrix<double> &a, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper map(pdom, Gd);
    double r = devL0_Norm2Sq(a, pdom, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static double devL2_MaxDiag(Matrix<double> &a, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper map(pdom, Gd);
    double r = devL0_MaxDiag(a, pdom, map, block_dim);
    if (cpm.size > 1) {
        CPML2_AllReduce(&r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
    return r;
}

}

#endif
