#ifndef FALM_MVL1_H
#define FALM_MVL1_H

#include "matrix.h"
#include "CPMBase.h"

namespace Falm {

class FalmMVDevCall {
public:
static Int reduction_buffer_size;
static Real *reduction_buffer_host, *reduction_buffer_device;

static void init() {
    // reduction_buffer_size = 0;
    // reduction_buffer_device = nullptr;
    // reduction_buffer_host = nullptr;
}

static void request_reduction_buffer(Int bufsize) {
    if (reduction_buffer_size < bufsize) {
        printf("reduction buffer enlarged from %d to %d\n", reduction_buffer_size, bufsize);
        if (reduction_buffer_size != 0) {
            falmErrCheckMacro(falmFreePinned(reduction_buffer_host));
            falmErrCheckMacro(falmFreeDevice(reduction_buffer_device));
        }
        falmErrCheckMacro(falmMallocPinned((void**)&reduction_buffer_host, bufsize * sizeof(Real)));
        falmErrCheckMacro(falmMallocDevice((void**)&reduction_buffer_device, bufsize * sizeof(Real)));
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

static void MV(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &ax, Region &pdm, const Region &map, dim3 block_dim, Stream stream = (Stream)0);

static Real DotProduct(Matrix<Real> &a, Matrix<Real> &b, Region &pdm, const Region &map, dim3 block_dim);

static Real EuclideanNormSq(Matrix<Real> &a, Region &pdm, const Region &map, dim3 block_dim);

// REAL L0Dev_MaxDiag(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim);

static Real MatColMax(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim);

static Real MatColMin(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim);

static Real MatColAbsMax(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim);

static Real MatColAbsMin(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim);

static Real VecMax(Matrix<Real> &a, Region &pdm, const Region &map, dim3 block_dim);

static Real VecMin(Matrix<Real> &a, Region &pdm, const Region &map, dim3 block_dim);

static void ScaleMatrix(Matrix<Real> &a, Real scale, dim3 block_dim);

static void MatrixAdd(Matrix<Real> &a, Matrix<Real> &b, dim3 block_dim);

static void Vecaxby(Real a, Matrix<Real> &x, Real b, Matrix<Real> &y, Matrix<Real> &result, dim3 block_dim);

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
