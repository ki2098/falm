#ifndef FALM_MVL1_H
#define FALM_MVL1_H

#include "matrix.h"
#include "mapper.h"

namespace Falm {

double L0Dev_DotProduct(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 block_dim);

double L0Dev_Norm2Sq(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 block_dim);

double L0Dev_MaxDiag(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 block_dim);

void L1Dev_ScaleMatrix(Matrix<double> &a, double scale, dim3 block_dim);

static inline double L1Dev_DotProduct(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    return L0Dev_DotProduct(a, b, pdom, map, block_dim);
}

static inline double L1Dev_Norm2Sq(Matrix<double> &a, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    return L0Dev_Norm2Sq(a, pdom, map, block_dim);
}

static inline double L1Dev_MaxDiag(Matrix<double> &a, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    return L0Dev_MaxDiag(a, pdom, map, block_dim);
}

}

#endif
