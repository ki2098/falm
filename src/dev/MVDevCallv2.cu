#include <cub/cub.cuh>
#include "../MVDevCallv2.h"
#include "devutil.cuh"

namespace Falm {

size_t FalmMVDevCallv2::tmp_result_size = 0;
size_t FalmMVDevCallv2::tmp_storage_size = 0;
void *FalmMVDevCallv2::tmp_storage = nullptr;
void *FalmMVDevCallv2::tmp_result = nullptr;

__global__ void kernel_MV(const MatrixFrame<Real> *va, const MatrixFrame<Real> *vx, const MatrixFrame<Real> *vax, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    const MatrixFrame<Real> &a=*va, &x=*vx, &ax=*vax;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idxc = IDX(i  , j  , k  , pdm_shape);
        Int idxe = IDX(i+1, j  , k  , pdm_shape);
        Int idxw = IDX(i-1, j  , k  , pdm_shape);
        Int idxn = IDX(i  , j+1, k  , pdm_shape);
        Int idxs = IDX(i  , j-1, k  , pdm_shape);
        Int idxt = IDX(i  , j  , k+1, pdm_shape);
        Int idxb = IDX(i  , j  , k-1, pdm_shape);
        Real ac = a(idxc, 0);
        Real aw = a(idxc, 1);
        Real ae = a(idxc, 2);
        Real as = a(idxc, 3);
        Real an = a(idxc, 4);
        Real ab = a(idxc, 5);
        Real at = a(idxc, 6);
        Real xc = x(idxc);
        Real xe = x(idxe);
        Real xw = x(idxw);
        Real xn = x(idxn);
        Real xs = x(idxs);
        Real xt = x(idxt);
        Real xb = x(idxb);
        ax(idxc) = ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb;
    }
}

void FalmMVDevCallv2::MV(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &ax, Region &pdm, const Region &map, dim3 block_dim, Stream stream) {
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

__global__ void kernel_DotProduct(const MatrixFrame<Real> *va, const MatrixFrame<Real> *vb, Real *tmp_result, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    const MatrixFrame<Real> &a=*va, &b=*vb;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    Real tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        tmp = a(idx) * b(idx);
    }
    tmp_result[tidx] = tmp;
}

Real FalmMVDevCallv2::DotProduct(Matrix<Real> &a, Matrix<Real> &b, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(Real));
    kernel_DotProduct<<<grid_dim, block_dim>>>(a.devptr, b.devptr, (Real*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    Real _result;
    cub::DeviceReduce::Sum(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Sum(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);

    return _result;
}

__global__ void kernel_EuclideanNormSq(const MatrixFrame<Real> *va, Real *tmp_result, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    Real tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        for (Int __d = 0; __d < a.shape[1]; __d ++) {
            tmp += a(idx, __d) * a(idx, __d);
        }
    }
    tmp_result[tidx] = tmp;
}

Real FalmMVDevCallv2::EuclideanNormSq(Matrix<Real> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(Real));
    kernel_EuclideanNormSq<<<grid_dim, block_dim>>>(a.devptr, (Real*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    Real _result;
    cub::DeviceReduce::Sum(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Sum(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);

    return _result;
}

__global__ void kernel_MatColMax(const MatrixFrame<Real> *va, Int col, Real *tmp_result, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    Real tmp = -FLT_MAX;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        tmp = a(idx, col);
    }
    tmp_result[tidx] = tmp;
}

__global__ void kernel_MatColMin(const MatrixFrame<Real> *va, Int col, Real *tmp_result, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    Real tmp = FLT_MAX;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        tmp = a(idx, col);
    }
    tmp_result[tidx] = tmp;
}

__global__ void kernel_MatColAbsMax(const MatrixFrame<Real> *va, Int col, Real *tmp_result, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    Real tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        tmp = fabs(a(idx, col));
    }
    tmp_result[tidx] = tmp;
}

__global__ void kernel_MatColAbsMin(const MatrixFrame<Real> *va, Int col, Real *tmp_result, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    Real tmp = FLT_MAX;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        tmp = fabs(a(idx, col));
    }
    tmp_result[tidx] = tmp;
}

Real FalmMVDevCallv2::MatColMax(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(Real));
    kernel_MatColMax<<<grid_dim, block_dim>>>(a.devptr, col, (Real*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    Real _result;
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);

    return _result;
}

Real FalmMVDevCallv2::MatColMin(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(Real));
    kernel_MatColMin<<<grid_dim, block_dim>>>(a.devptr, col, (Real*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    Real _result;
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);

    return _result;
}

Real FalmMVDevCallv2::MatColAbsMax(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(Real));
    kernel_MatColAbsMax<<<grid_dim, block_dim>>>(a.devptr, col, (Real*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    Real _result;
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);

    return _result;
}

Real FalmMVDevCallv2::MatColAbsMin(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(Real));
    kernel_MatColAbsMin<<<grid_dim, block_dim>>>(a.devptr, col, (Real*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    Real _result;
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);

    return _result;
}

__global__ void kernel_VecMax(const MatrixFrame<Real> *va, Real *tmp_result, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    Real tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        for (Int n = 0; n < a.shape[1]; n ++) {
            tmp += a(idx, n) * a(idx, n);
        }
        tmp = sqrt(tmp);
    }
    tmp_result[tidx] = tmp;
}

__global__ void kernel_VecMin(const MatrixFrame<Real> *va, Real *tmp_result, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    size_t tidx = GLOBAL_THREAD_IDX();
    Real tmp = FLT_MAX;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        for (Int n = 0; n < a.shape[1]; n ++) {
            tmp += a(idx, n) * a(idx, n);
        }
        tmp = sqrt(tmp);
    }
    tmp_result[tidx] = tmp;
}

Real FalmMVDevCallv2::VecMax(Matrix<Real> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(Real));
    kernel_VecMax<<<grid_dim, block_dim>>>(a.devptr, (Real*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    Real _result;
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Max(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);

    return _result;
}

Real FalmMVDevCallv2::VecMin(Matrix<Real> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int gnthreads = PRODUCT3(block_dim)*PRODUCT3(grid_dim);
    request_tmp_result(gnthreads*sizeof(Real));
    kernel_VecMin<<<grid_dim, block_dim>>>(a.devptr, (Real*)tmp_result, pdm.shape, map.shape, map.offset);

    size_t _tmp_storage_byte = 0;
    Real _result;
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);
    request_tmp_storage(_tmp_storage_byte);
    cub::DeviceReduce::Min(tmp_storage, _tmp_storage_byte, (Real*)tmp_result, &_result, gnthreads);

    return _result;
}

__global__ void kernel_ScaleMatrix(const MatrixFrame<Real> *va, Real scale) {
    const MatrixFrame<Real> &a=*va;
    Int tidx  = IDX(threadIdx, blockDim);
    Int bidx  = IDX(blockIdx, gridDim);
    Int bsize = PRODUCT3(blockDim);
    Int gtidx = tidx + bidx * bsize;
    if (gtidx < a.size) {
        a(gtidx) *= scale;
    }
}

void FalmMVDevCallv2::ScaleMatrix(Matrix<Real> &a, Real scale, dim3 block_dim) {
    Int n_threads = PRODUCT3(block_dim);
    Int n_blocks = (a.size + n_threads - 1) / n_threads;
    kernel_ScaleMatrix<<<n_blocks, n_threads>>>(a.devptr, scale);
    falmWaitStream();
}

__global__ void kernel_MatrixAdd(const MatrixFrame<Real> *va, const MatrixFrame<Real> *vb) {
    const MatrixFrame<Real> &a=*va, &b=*vb;
    Int tidx  = IDX(threadIdx, blockDim);
    Int bidx  = IDX(blockIdx, gridDim);
    Int bsize = PRODUCT3(blockDim);
    Int gtidx = tidx + bidx * bsize;
    if (gtidx < a.size) {
        a(gtidx) += b(gtidx);
    }
}

void FalmMVDevCallv2::MatrixAdd(Matrix<Real> &a, Matrix<Real> &b, dim3 block_dim) {
    Int n_threads = PRODUCT3(block_dim);
    Int n_blocks = (a.size + n_threads - 1) / n_threads;
    kernel_MatrixAdd<<<n_blocks, n_threads>>>(a.devptr, b.devptr);
    falmWaitStream();
}

}