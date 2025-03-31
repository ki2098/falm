#include <cub/cub.cuh>
#include "../MVDevCallv2.h"
#include "devutil.cuh"

namespace Falm {

size_t FalmMVDevCallv2::tmp_result_size = 0;
size_t FalmMVDevCallv2::tmp_storage_size = 0;
void *FalmMVDevCallv2::tmp_storage = nullptr;
void *FalmMVDevCallv2::tmp_result = nullptr;

__global__ void kernel_MV(const MatrixFrame<REAL> *va, const MatrixFrame<REAL> *vx, const MatrixFrame<REAL> *vax, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va, &x=*vx, &ax=*vax;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxc = IDX(i  , j  , k  , pdm_shape);
        INT idxe = IDX(i+1, j  , k  , pdm_shape);
        INT idxw = IDX(i-1, j  , k  , pdm_shape);
        INT idxn = IDX(i  , j+1, k  , pdm_shape);
        INT idxs = IDX(i  , j-1, k  , pdm_shape);
        INT idxt = IDX(i  , j  , k+1, pdm_shape);
        INT idxb = IDX(i  , j  , k-1, pdm_shape);
        REAL ac = a(idxc, 0);
        REAL aw = a(idxc, 1);
        REAL ae = a(idxc, 2);
        REAL as = a(idxc, 3);
        REAL an = a(idxc, 4);
        REAL ab = a(idxc, 5);
        REAL at = a(idxc, 6);
        REAL xc = x(idxc);
        REAL xe = x(idxe);
        REAL xw = x(idxw);
        REAL xn = x(idxn);
        REAL xs = x(idxs);
        REAL xt = x(idxt);
        REAL xb = x(idxb);
        ax(idxc) = ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb;
    }
}

void FalmMVDevCallv2::MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Region &pdm, const Region &map, dim3 block_dim, STREAM stream) {
    assert(
        a.shape[0] == x.shape[0] && a.shape[0] == ax.shape[0] &&
        a.shape[1] == 7 && x.shape[1] == 1 && ax.shape[1] == 1
    );
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );

    kernel_MV<<<grid_dim, block_dim, 0, stream>>>(a.devptr, x.devptr, ax.devptr, pdm.shape, map.shape, map.offset);
}

__global__ void kernel_DotProduct(const MatrixFrame<REAL> *va, const MatrixFrame<REAL> *vb, REAL *tmp_result, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va, &b=*vb;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    REAL tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        tmp = a(idx) * b(idx);
    }
    tmp_result[tidx] = tmp;
}

REAL FalmMVDevCallv2::DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(REAL));
    kernel_DotProduct<<<grid_dim, block_dim>>>(a.devptr, b.devptr, (REAL*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    REAL _result;
    cub::DeviceReduce::Sum(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Sum(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);

    return _result;
}

__global__ void kernel_EuclideanNormSq(const MatrixFrame<REAL> *va, REAL *tmp_result, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    REAL tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        for (INT __d = 0; __d < a.shape[1]; __d ++) {
            tmp += a(idx, __d) * a(idx, __d);
        }
    }
    tmp_result[tidx] = tmp;
}

REAL FalmMVDevCallv2::EuclideanNormSq(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(REAL));
    kernel_EuclideanNormSq<<<grid_dim, block_dim>>>(a.devptr, (REAL*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    REAL _result;
    cub::DeviceReduce::Sum(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Sum(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);

    return _result;
}

__global__ void kernel_MatColMax(const MatrixFrame<REAL> *va, INT col, REAL *tmp_result, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    REAL tmp = -FLT_MAX;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        tmp = a(idx, col);
    }
    tmp_result[tidx] = tmp;
}

__global__ void kernel_MatColMin(const MatrixFrame<REAL> *va, INT col, REAL *tmp_result, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    REAL tmp = FLT_MAX;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        tmp = a(idx, col);
    }
    tmp_result[tidx] = tmp;
}

__global__ void kernel_MatColAbsMax(const MatrixFrame<REAL> *va, INT col, REAL *tmp_result, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    REAL tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        tmp = fabs(a(idx, col));
    }
    tmp_result[tidx] = tmp;
}

__global__ void kernel_MatColAbsMin(const MatrixFrame<REAL> *va, INT col, REAL *tmp_result, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    REAL tmp = FLT_MAX;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        tmp = fabs(a(idx, col));
    }
    tmp_result[tidx] = tmp;
}

REAL FalmMVDevCallv2::MatColMax(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(REAL));
    kernel_MatColMax<<<grid_dim, block_dim>>>(a.devptr, col, (REAL*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    REAL _result;
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);

    return _result;
}

REAL FalmMVDevCallv2::MatColMin(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(REAL));
    kernel_MatColMin<<<grid_dim, block_dim>>>(a.devptr, col, (REAL*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    REAL _result;
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);

    return _result;
}

REAL FalmMVDevCallv2::MatColAbsMax(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(REAL));
    kernel_MatColAbsMax<<<grid_dim, block_dim>>>(a.devptr, col, (REAL*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    REAL _result;
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);

    return _result;
}

REAL FalmMVDevCallv2::MatColAbsMin(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(REAL));
    kernel_MatColAbsMin<<<grid_dim, block_dim>>>(a.devptr, col, (REAL*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    REAL _result;
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);

    return _result;
}

__global__ void kernel_VecMax(const MatrixFrame<REAL> *va, REAL *tmp_result, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    REAL tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        for (INT n = 0; n < a.shape[1]; n ++) {
            tmp += a(idx, n) * a(idx, n);
        }
        tmp = sqrt(tmp);
    }
    tmp_result[tidx] = tmp;
}

__global__ void kernel_VecMin(const MatrixFrame<REAL> *va, REAL *tmp_result, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    REAL tmp = FLT_MAX;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        for (INT n = 0; n < a.shape[1]; n ++) {
            tmp += a(idx, n) * a(idx, n);
        }
        tmp = sqrt(tmp);
    }
    tmp_result[tidx] = tmp;
}

REAL FalmMVDevCallv2::VecMax(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(REAL));
    kernel_VecMax<<<grid_dim, block_dim>>>(a.devptr, (REAL*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    REAL _result;
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);

    return _result;
}

REAL FalmMVDevCallv2::VecMin(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(REAL));
    kernel_VecMin<<<grid_dim, block_dim>>>(a.devptr, (REAL*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    REAL _result;
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (REAL*)tmp_result, &_result, gnthreads);

    return _result;
}

__global__ void kernel_ScaleMatrix(const MatrixFrame<REAL> *va, REAL scale) {
    const MatrixFrame<REAL> &a=*va;
    INT tidx  = IDX(threadIdx, blockDim);
    INT bidx  = IDX(blockIdx, gridDim);
    INT bsize = PRODUCT3(blockDim);
    INT gtidx = tidx + bidx * bsize;
    if (gtidx < a.size) {
        a(gtidx) *= scale;
    }
}

void FalmMVDevCallv2::ScaleMatrix(Matrix<REAL> &a, REAL scale, dim3 block_dim) {
    INT n_threads = PRODUCT3(block_dim);
    INT n_blocks = (a.size + n_threads - 1) / n_threads;
    kernel_ScaleMatrix<<<n_blocks, n_threads>>>(a.devptr, scale);
    falmWaitStream();
}

__global__ void kernel_MatrixAdd(const MatrixFrame<REAL> *va, const MatrixFrame<REAL> *vb) {
    const MatrixFrame<REAL> &a=*va, &b=*vb;
    INT tidx  = IDX(threadIdx, blockDim);
    INT bidx  = IDX(blockIdx, gridDim);
    INT bsize = PRODUCT3(blockDim);
    INT gtidx = tidx + bidx * bsize;
    if (gtidx < a.size) {
        a(gtidx) += b(gtidx);
    }
}

void FalmMVDevCallv2::MatrixAdd(Matrix<REAL> &a, Matrix<REAL> &b, dim3 block_dim) {
    INT n_threads = PRODUCT3(block_dim);
    INT n_blocks = (a.size + n_threads - 1) / n_threads;
    kernel_MatrixAdd<<<n_blocks, n_threads>>>(a.devptr, b.devptr);
    falmWaitStream();
}

}