#ifndef FALM_MVL1_H
#define FALM_MVL1_H

#include "matrix.h"
#include "mapper.h"

namespace Falm {

double devL0_DotProduct(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 block_dim);

double devL0_Norm2Sq(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 block_dim);

double devL0_MaxDiag(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 block_dim);

void devL1_ScaleMatrix(Matrix<double> &a, double scale, dim3 block_dim);

static inline double devL1_DotProduct(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    return devL0_DotProduct(a, b, pdom, map, block_dim);
}

static inline double devL1_Norm2Sq(Matrix<double> &a, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    return devL0_Norm2Sq(a, pdom, map, block_dim);
}

static inline double devL1_MaxDiag(Matrix<double> &a, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    return devL0_MaxDiag(a, pdom, map, block_dim);
}

}

#endif
