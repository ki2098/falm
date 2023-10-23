#ifndef FALM_MVL1_H
#define FALM_MVL1_H

#include "matrix.h"
#include "CPMBase.h"

namespace Falm {

REAL L0Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, Region &pdm, const Region &map, dim3 block_dim);

REAL L0Dev_EuclideanNormSq(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

REAL L0Dev_MaxDiag(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

void L1Dev_ScaleMatrix(Matrix<REAL> &a, REAL scale, dim3 block_dim);

static inline REAL L1Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    return L0Dev_DotProduct(a, b, pdm, map, block_dim);
}

static inline REAL L1Dev_EuclideanNormSq(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    return L0Dev_EuclideanNormSq(a, pdm, map, block_dim);
}

static inline REAL L1Dev_MaxDiag(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    return L0Dev_MaxDiag(a, pdm, map, block_dim);
}

}

#endif
