#ifndef FALM_MVL1_H
#define FALM_MVL1_H

#include "matrix.h"
#include "CPMBase.h"

namespace Falm {

class FalmMVDevCall {
public:
static INT reduction_buffer_size;
static REAL *reduction_buffer_host, *reduction_buffer_device;

static void init() {
    // reduction_buffer_size = 0;
    // reduction_buffer_device = nullptr;
    // reduction_buffer_host = nullptr;
}

static void request_reduction_buffer(INT bufsize) {
    if (reduction_buffer_size < bufsize) {
        printf("reduction buffer enlarged from %d to %d\n", reduction_buffer_size, bufsize);
        if (reduction_buffer_size != 0) {
            falmErrCheckMacro(falmFreePinned(reduction_buffer_host));
            falmErrCheckMacro(falmFreeDevice(reduction_buffer_device));
        }
        falmErrCheckMacro(falmMallocPinned((void**)&reduction_buffer_host, bufsize * sizeof(REAL)));
        falmErrCheckMacro(falmMallocDevice((void**)&reduction_buffer_device, bufsize * sizeof(REAL)));
        reduction_buffer_size = bufsize;
    }
}

static void release() {
    if (reduction_buffer_size > 0) {
        falmErrCheckMacro(falmFreePinned(reduction_buffer_host));
        falmErrCheckMacro(falmFreeDevice(reduction_buffer_device));
    }
    reduction_buffer_size = 0;
}

static void MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Region &pdm, const Region &map, dim3 block_dim, STREAM stream = (STREAM)0);

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

static void MatrixAdd(Matrix<REAL> &a, Matrix<REAL> &b, dim3 block_dim);

static void Vecaxby(REAL a, Matrix<REAL> &x, REAL b, Matrix<REAL> &y, Matrix<REAL> &result, dim3 block_dim);

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
