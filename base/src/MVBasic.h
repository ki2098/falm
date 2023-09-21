#ifndef FALM_MVBASIC_H
#define FALM_MVBASIC_H

#include "MVBasicDev.h"
#include "CPM.h"

namespace Falm {

static double devL2_dotProduct(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm) {
    double r = devL1_DotProduct(a, b, pdom, map, block_dim);
    if (cpm.size > 1) {
        CPML1_AllReduce(&r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static double devL2_Norm2Sq(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm) {
    double r = devL1_Norm2Sq(a, pdom, map, block_dim);
    if (cpm.size > 1) {
        CPML1_AllReduce(&r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static double devL2_MaxDiag(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm) {
    double r = devL1_MaxDiag(a, pdom, map, block_dim);
    if (cpm.size > 1) {
        CPML1_AllReduce(&r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
    return r;
}

}

#endif
