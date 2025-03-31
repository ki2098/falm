#ifndef FALM_MVDEVCALLV2_H
#define FALM_MVDEVCALLV2_H

#include "matrix.h"
#include "CPMBase.h"

namespace Falm {

class FalmMVDevCallv2 {

public:

static void *tmp_storage;
static size_t tmp_storage_size;
static void *tmp_result;
static size_t tmp_result_size;

static void init() {}

static void request_tmp_storage(size_t size) {
    if (tmp_storage_size < size) {
        if (tmp_storage_size != 0) {
            falmErrCheckMacro(falmFreeDevice(tmp_storage));
        }
        falmErrCheckMacro(falmMallocDevice(&tmp_storage, size));
        tmp_storage_size = size;
    }
}

static void request_tmp_result(size_t size) {
    if (tmp_result_size < size) {
        if (tmp_result_size != 0) {
            falmErrCheckMacro(falmFreeDevice(tmp_result));
        }
        falmErrCheckMacro(falmMallocDevice(&tmp_result, size));
        tmp_result_size = size;
    }
}

static void release() {
    if (tmp_storage_size > 0) {
        falmErrCheckMacro(falmFreeDevice(tmp_storage));
        tmp_storage_size = 0;
        tmp_storage = nullptr;
    }
    if (tmp_result_size > 0) {
        falmErrCheckMacro(falmFreeDevice(tmp_result));
        tmp_result_size = 0;
        tmp_result = nullptr;
    }
}

static void MV(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &ax, Region &pdm, const Region &map, dim3 block_dim, Stream stream = 0);

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

};

}

#endif