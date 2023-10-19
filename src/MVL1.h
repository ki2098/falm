#ifndef FALM_MVL1_H
#define FALM_MVL1_H

#include "matrix.h"
#include "mapper.h"

namespace Falm {

REAL L0Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, Mapper &pdm, const Mapper &map, dim3 block_dim);

REAL L0Dev_EuclideanNormSq(Matrix<REAL> &a, Mapper &pdm, const Mapper &map, dim3 block_dim);

REAL L0Dev_MaxDiag(Matrix<REAL> &a, Mapper &pdm, const Mapper &map, dim3 block_dim);

void L1Dev_ScaleMatrix(Matrix<REAL> &a, REAL scale, dim3 block_dim);

static inline REAL L1Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, Mapper &pdm, INT gc, dim3 block_dim) {
    Mapper map(pdm, gc);
    return L0Dev_DotProduct(a, b, pdm, map, block_dim);
}

static inline REAL L1Dev_EuclideanNormSq(Matrix<REAL> &a, Mapper &pdm, INT gc, dim3 block_dim) {
    Mapper map(pdm, gc);
    return L0Dev_EuclideanNormSq(a, pdm, map, block_dim);
}

static inline REAL L1Dev_MaxDiag(Matrix<REAL> &a, Mapper &pdm, INT gc, dim3 block_dim) {
    Mapper map(pdm, gc);
    return L0Dev_MaxDiag(a, pdm, map, block_dim);
}

}

#endif
