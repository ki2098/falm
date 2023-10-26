#ifndef FALM_MVL1_H
#define FALM_MVL1_H

#include "matrix.h"
#include "CPMBase.h"

namespace Falm {

REAL L0Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, Region &pdm, const Region &map, dim3 block_dim);

REAL L0Dev_EuclideanNormSq(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

// REAL L0Dev_MaxDiag(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

REAL L0Dev_MatColMax(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim);

REAL L0Dev_MatColMin(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim);

REAL L0Dev_MatColAbsMax(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim);

REAL L0Dev_MatColAbsMin(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim);

REAL L0Dev_VecMax(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

REAL L0Dev_VecMin(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

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
    return L0Dev_MatColAbsMax(a, 0, pdm, map, block_dim);
}

static inline REAL L1Dev_MatColMax(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    return L0Dev_MatColMax(a, col, pdm, map, block_dim);
}

static inline REAL L1Dev_MatColMin(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    return L0Dev_MatColMin(a, col, pdm, map, block_dim);
}

static inline REAL L1Dev_MatColAbsMax(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    return L0Dev_MatColAbsMax(a, col, pdm, map, block_dim);
}

static inline REAL L1Dev_MatColAbsMin(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    return L0Dev_MatColAbsMin(a, col, pdm, map, block_dim);
}

static inline REAL L1Dev_VecMax(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    return L0Dev_VecMax(a, pdm, map, block_dim);
}

static inline REAL L1Dev_VecMin(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    return L0Dev_VecMin(a, pdm, map, block_dim);
}

}

#endif
