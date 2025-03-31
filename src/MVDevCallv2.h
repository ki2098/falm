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

static void MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Region &pdm, const Region &map, dim3 block_dim, STREAM stream = 0);

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

};

}

#endif