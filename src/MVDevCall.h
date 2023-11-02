#ifndef FALM_MVL1_H
#define FALM_MVL1_H

#include "matrix.h"
#include "CPMBase.h"

namespace Falm {

class MVDevCall {
public:

static void MVMult(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Region &pdm, const Region &map, dim3 block_dim, STREAM stream = (STREAM)0);

static REAL DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, Region &pdm, const Region &map, dim3 block_dim);

static REAL EuclideanNormSq(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

// REAL L0Dev_MaxDiag(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

static REAL MatColMax(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim);

static REAL MatColMin(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim);

static REAL MatColAbsMax(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim);

static REAL MatColAbsMin(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim);

static REAL VecMax(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

static REAL VecMin(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

static void ScaleMatrix(Matrix<REAL> &a, REAL scale, dim3 block_dim);

// static inline REAL L1Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     return DotProduct(a, b, pdm, map, block_dim);
// }

// static inline REAL L1Dev_EuclideanNormSq(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     return EuclideanNormSq(a, pdm, map, block_dim);
// }

// static inline REAL L1Dev_MaxDiag(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     return MatColAbsMax(a, 0, pdm, map, block_dim);
// }

// static inline REAL L1Dev_MatColMax(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     return MatColMax(a, col, pdm, map, block_dim);
// }

// static inline REAL L1Dev_MatColMin(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     return MatColMin(a, col, pdm, map, block_dim);
// }

// static inline REAL L1Dev_MatColAbsMax(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     return MatColAbsMax(a, col, pdm, map, block_dim);
// }

// static inline REAL L1Dev_MatColAbsMin(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     return MatColAbsMin(a, col, pdm, map, block_dim);
// }

// static inline REAL L1Dev_VecMax(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     return VecMax(a, pdm, map, block_dim);
// }

// static inline REAL L1Dev_VecMin(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     return VecMin(a, pdm, map, block_dim);
// }

};

}

#endif
