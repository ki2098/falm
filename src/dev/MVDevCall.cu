#include "../MVDevCall.h"
#include "devutil.cuh"

namespace Falm {

Int   FalmMVDevCall::reduction_buffer_size = 0;
Real *FalmMVDevCall::reduction_buffer_device = nullptr;
Real *FalmMVDevCall::reduction_buffer_host = nullptr;

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

void FalmMVDevCall::MV(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &ax, Region &pdm, const Region &map, dim3 block_dim, Stream stream) {
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

__global__ void kernel_DotProduct(const MatrixFrame<Real> *va, const MatrixFrame<Real> *vb, Real *partial_sum_dev, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    extern __shared__ Real cache[];
    const MatrixFrame<Real> &a=*va, &b=*vb;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    Int tidx = IDX(threadIdx, blockDim);
    Real tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        tmp = a(idx) * b(idx);
    }
    cache[tidx] = tmp;
    __syncthreads();

    Int length = PRODUCT3(blockDim);
    while (length > 1) {
        Int cut = length / 2;
        Int reduce = length - cut;
        if (tidx < cut) {
            cache[tidx] += cache[tidx + reduce];
        }
        __syncthreads();
        length = reduce;
    }

    if (tidx == 0) {
        partial_sum_dev[IDX(blockIdx, gridDim)] = cache[0];
    }
}

Real FalmMVDevCall::DotProduct(Matrix<Real> &a, Matrix<Real> &b, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int n_blocks = PRODUCT3(grid_dim);
    Int n_threads = PRODUCT3(block_dim);
    // REAL *partial_sum,*partial_sum_dev;
    // falmErrCheckMacro(falmMalloc((void**)&partial_sum, sizeof(REAL) * n_blocks));
    // falmErrCheckMacro(falmMallocDevice((void**)&partial_sum_dev, sizeof(REAL) * n_blocks));
    request_reduction_buffer(n_blocks);
    size_t shared_size = n_threads * sizeof(Real);

    kernel_DotProduct<<<grid_dim, block_dim, shared_size>>>(a.devptr, b.devptr, reduction_buffer_device, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(reduction_buffer_host, reduction_buffer_device, sizeof(Real) * n_blocks, MCP::Dev2Hst));
    Real sum = reduction_buffer_host[0];
    for (Int i = 1; i < n_blocks; i ++) {
        sum += reduction_buffer_host[i];
    }

    // falmErrCheckMacro(falmFree(partial_sum));
    // falmErrCheckMacro(falmFreeDevice(partial_sum_dev));

    return sum;
}

__global__ void kernel_EuclideanNormSq(const MatrixFrame<Real> *va, Real *partial_sum_dev, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    extern __shared__ Real cache[];
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    Int tidx = IDX(threadIdx, blockDim);
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
    cache[tidx] = tmp;
    __syncthreads();

    Int length = PRODUCT3(blockDim);
    while (length > 1) {
        Int cut = length / 2;
        Int reduce = length - cut;
        if (tidx < cut) {
            cache[tidx] += cache[tidx + reduce];
        }
        __syncthreads();
        length = reduce;
    }

    if (tidx == 0) {
        partial_sum_dev[IDX(blockIdx, gridDim)] = cache[0];
    }
}

Real FalmMVDevCall::EuclideanNormSq(Matrix<Real> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int n_blocks = PRODUCT3(grid_dim);
    Int n_threads = PRODUCT3(block_dim);
    // REAL *partial_sum,*partial_sum_dev;
    // falmErrCheckMacro(falmMalloc((void**)&partial_sum, sizeof(REAL) * n_blocks));
    // falmErrCheckMacro(falmMallocDevice((void**)&partial_sum_dev, sizeof(REAL) * n_blocks));
    request_reduction_buffer(n_blocks);
    size_t shared_size = n_threads * sizeof(Real);

    kernel_EuclideanNormSq<<<grid_dim, block_dim, shared_size>>>(a.devptr, reduction_buffer_device, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(reduction_buffer_host, reduction_buffer_device, sizeof(Real) * n_blocks, MCP::Dev2Hst));
    Real sum = reduction_buffer_host[0];
    for (Int i = 1; i < n_blocks; i ++) {
        sum += reduction_buffer_host[i];
    }

    // falmErrCheckMacro(falmFree(partial_sum));
    // falmErrCheckMacro(falmFreeDevice(partial_sum_dev));

    return sum;
}

/* __global__ void kernel_MaxDiag(const MatrixFrame<REAL> *va, REAL *partial_max_dev, INTx3 pdm_shape, INTx3 map_shape, INTx3 map_offset) {
    extern __shared__ REAL cache[];
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    INT tidx = IDX(threadIdx, blockDim);
    REAL tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        tmp = fabs(a(idx, 0));
    }
    cache[tidx] = tmp;
    __syncthreads();

    INT length = PRODUCT3(blockDim);
    while (length > 1) {
        INT cut = length / 2;
        INT reduce = length - cut;
        if (tidx < cut) {
            if (cache[tidx + reduce] > cache[tidx]) {
                cache[tidx] = cache[tidx + reduce];
            }
        }
        __syncthreads();
        length = reduce;
    }

    if (tidx == 0) {
        partial_max_dev[IDX(blockIdx, gridDim)] = cache[0];
    }
} */

__global__ void kernel_MatColMax(const MatrixFrame<Real> *va, Int col, Real *partial_max_dev, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    extern __shared__ Real cache[];
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    Int tidx = IDX(threadIdx, blockDim);
    Real tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        tmp = a(idx, col);
    }
    cache[tidx] = tmp;
    __syncthreads();

    Int length = PRODUCT3(blockDim);
    while (length > 1) {
        Int cut = length / 2;
        Int reduce = length - cut;
        if (tidx < cut) {
            if (cache[tidx + reduce] > cache[tidx]) {
                cache[tidx] = cache[tidx + reduce];
            }
        }
        __syncthreads();
        length = reduce;
    }

    if (tidx == 0) {
        partial_max_dev[IDX(blockIdx, gridDim)] = cache[0];
    }
}

__global__ void kernel_MatColMin(const MatrixFrame<Real> *va, Int col, Real *partial_max_dev, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    extern __shared__ Real cache[];
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    Int tidx = IDX(threadIdx, blockDim);
    Real tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        tmp = a(idx, col);
    }
    cache[tidx] = tmp;
    __syncthreads();

    Int length = PRODUCT3(blockDim);
    while (length > 1) {
        Int cut = length / 2;
        Int reduce = length - cut;
        if (tidx < cut) {
            if (cache[tidx + reduce] < cache[tidx]) {
                cache[tidx] = cache[tidx + reduce];
            }
        }
        __syncthreads();
        length = reduce;
    }

    if (tidx == 0) {
        partial_max_dev[IDX(blockIdx, gridDim)] = cache[0];
    }
}

__global__ void kernel_MatColAbsMax(const MatrixFrame<Real> *va, Int col, Real *partial_max_dev, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    extern __shared__ Real cache[];
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    Int tidx = IDX(threadIdx, blockDim);
    Real tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        tmp = fabs(a(idx, col));
    }
    cache[tidx] = tmp;
    // printf("%lf\n", tmp);
    __syncthreads();

    Int length = PRODUCT3(blockDim);
    while (length > 1) {
        Int cut = length / 2;
        Int reduce = length - cut;
        if (tidx < cut) {
            if (cache[tidx + reduce] > cache[tidx]) {
                cache[tidx] = cache[tidx + reduce];
            }
        }
        __syncthreads();
        length = reduce;
    }

    if (tidx == 0) {
        partial_max_dev[IDX(blockIdx, gridDim)] = cache[0];
    }
}

__global__ void kernel_MatColAbsMin(const MatrixFrame<Real> *va, Int col, Real *partial_max_dev, Int3 pdm_shape, Int3 map_shape, Int3 map_offset) {
    extern __shared__ Real cache[];
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    Int tidx = IDX(threadIdx, blockDim);
    Real tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        tmp = fabs(a(idx, col));
    }
    cache[tidx] = tmp;
    __syncthreads();

    Int length = PRODUCT3(blockDim);
    while (length > 1) {
        Int cut = length / 2;
        Int reduce = length - cut;
        if (tidx < cut) {
            if (cache[tidx + reduce] < cache[tidx]) {
                cache[tidx] = cache[tidx + reduce];
            }
        }
        __syncthreads();
        length = reduce;
    }

    if (tidx == 0) {
        partial_max_dev[IDX(blockIdx, gridDim)] = cache[0];
    }
}

/* REAL L0Dev_MaxDiag(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    REAL *partial_max_dev = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_MaxDiag<<<grid_dim, block_dim, shared_size>>>(a.devptr, partial_max_dev, pdm.shape, map.shape, map.offset);

    falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCpType::Dev2Hst);
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] > maximum) {
            maximum = partial_max[i];
        }
    }

    falmFreePinned(partial_max);
    falmFreeDevice(partial_max_dev);

    return maximum;
} */

Real FalmMVDevCall::MatColMax(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int n_blocks = PRODUCT3(grid_dim);
    Int n_threads = PRODUCT3(block_dim);
    // REAL *partial_max,*partial_max_dev;
    // falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    // falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    request_reduction_buffer(n_blocks);
    size_t shared_size = n_threads * sizeof(Real);

    kernel_MatColMax<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, reduction_buffer_device, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(reduction_buffer_host, reduction_buffer_device, sizeof(Real) * n_blocks, MCP::Dev2Hst));
    Real maximum = reduction_buffer_host[0];
    for (Int i = 1; i < n_blocks; i ++) {
        if (reduction_buffer_host[i] > maximum) {
            maximum = reduction_buffer_host[i];
        }
    }

    // falmErrCheckMacro(falmFree(partial_max));
    // falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
}

Real FalmMVDevCall::MatColMin(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int n_blocks = PRODUCT3(grid_dim);
    Int n_threads = PRODUCT3(block_dim);
    // REAL *partial_max,*partial_max_dev;
    // falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    // falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    request_reduction_buffer(n_blocks);
    size_t shared_size = n_threads * sizeof(Real);

    kernel_MatColMin<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, reduction_buffer_device, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(reduction_buffer_host, reduction_buffer_device, sizeof(Real) * n_blocks, MCP::Dev2Hst));
    Real maximum = reduction_buffer_host[0];
    for (Int i = 1; i < n_blocks; i ++) {
        if (reduction_buffer_host[i] < maximum) {
            maximum = reduction_buffer_host[i];
        }
    }

    // falmErrCheckMacro(falmFree(partial_max));
    // falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
}

Real FalmMVDevCall::MatColAbsMax(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int n_blocks = PRODUCT3(grid_dim);
    Int n_threads = PRODUCT3(block_dim);
    // REAL *partial_max,*partial_max_dev;
    // falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    // falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    request_reduction_buffer(n_blocks);
    size_t shared_size = n_threads * sizeof(Real);
    // printf("%d %d %d, %d %d %d, %d\n", block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z, shared_size);

    kernel_MatColAbsMax<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, reduction_buffer_device, pdm.shape, map.shape, map.offset);
    // falmWaitStream();
    falmErrCheckMacro(falmMemcpy(reduction_buffer_host, reduction_buffer_device, sizeof(Real) * n_blocks, MCP::Dev2Hst));
    Real maximum = reduction_buffer_host[0];
    for (Int i = 1; i < n_blocks; i ++) {
        if (reduction_buffer_host[i] > maximum) {
            maximum = reduction_buffer_host[i];
        }
    }

    // falmErrCheckMacro(falmFree(partial_max));
    // falmErrCheckMacro(falmFreeDevice(partial_max_dev));
    // printf("%e\n", maximum);
    return maximum;
}

Real FalmMVDevCall::MatColAbsMin(Matrix<Real> &a, Int col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int n_blocks = PRODUCT3(grid_dim);
    Int n_threads = PRODUCT3(block_dim);
    // REAL *partial_max,*partial_max_dev;
    // falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    // falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    request_reduction_buffer(n_blocks);
    size_t shared_size = n_threads * sizeof(Real);

    kernel_MatColAbsMin<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, reduction_buffer_device, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(reduction_buffer_host, reduction_buffer_device, sizeof(Real) * n_blocks, MCP::Dev2Hst));
    Real maximum = reduction_buffer_host[0];
    for (Int i = 1; i < n_blocks; i ++) {
        if (reduction_buffer_host[i] < maximum) {
            maximum = reduction_buffer_host[i];
        }
    }

    // falmErrCheckMacro(falmFree(partial_max));
    // falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
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

void FalmMVDevCall::ScaleMatrix(Matrix<Real> &a, Real scale, dim3 block_dim) {
    Int n_threads = PRODUCT3(block_dim);
    Int n_blocks = (a.size + n_threads - 1) / n_threads;
    kernel_ScaleMatrix<<<n_blocks, n_threads>>>(a.devptr, scale);
    falmWaitStream();
}

__global__ void kernel_VecMax(const MatrixFrame<Real> *va, Real *partial_max_dev, Int3 pdm_shape, Int3 map_shape, Int3 map_offset ) {
    extern __shared__ Real cache[];
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    Int tidx = IDX(threadIdx, blockDim);
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
    cache[tidx] = tmp;
    __syncthreads();

    Int length = PRODUCT3(blockDim);
    while (length > 1) {
        Int cut = length / 2;
        Int reduce = length - cut;
        if (tidx < cut) {
            if (cache[tidx + reduce] > cache[tidx]) {
                cache[tidx] = cache[tidx + reduce];
            }
        }
        __syncthreads();
        length = reduce;
    }

    if (tidx == 0) {
        partial_max_dev[IDX(blockIdx, gridDim)] = cache[0];
    }
}

__global__ void kernel_VecMin(const MatrixFrame<Real> *va, Real *partial_max_dev, Int3 pdm_shape, Int3 map_shape, Int3 map_offset ) {
    extern __shared__ Real cache[];
    const MatrixFrame<Real> &a=*va;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    Int tidx = IDX(threadIdx, blockDim);
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
    cache[tidx] = tmp;
    __syncthreads();

    Int length = PRODUCT3(blockDim);
    while (length > 1) {
        Int cut = length / 2;
        Int reduce = length - cut;
        if (tidx < cut) {
            if (cache[tidx + reduce] < cache[tidx]) {
                cache[tidx] = cache[tidx + reduce];
            }
        }
        __syncthreads();
        length = reduce;
    }

    if (tidx == 0) {
        partial_max_dev[IDX(blockIdx, gridDim)] = cache[0];
    }
}

Real FalmMVDevCall::VecMax(Matrix<Real> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int n_blocks = PRODUCT3(grid_dim);
    Int n_threads = PRODUCT3(block_dim);
    // REAL *partial_max,*partial_max_dev;
    // falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    // falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    request_reduction_buffer(n_blocks);
    size_t shared_size = n_threads * sizeof(Real);

    kernel_VecMax<<<grid_dim, block_dim, shared_size>>>(a.devptr, reduction_buffer_device, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(reduction_buffer_host, reduction_buffer_device, sizeof(Real) * n_blocks, MCP::Dev2Hst));
    Real maximum = reduction_buffer_host[0];
    for (Int i = 1; i < n_blocks; i ++) {
        if (reduction_buffer_host[i] > maximum) {
            maximum = reduction_buffer_host[i];
        }
    }

    // falmErrCheckMacro(falmFree(partial_max));
    // falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
}

Real FalmMVDevCall::VecMin(Matrix<Real> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int n_blocks = PRODUCT3(grid_dim);
    Int n_threads = PRODUCT3(block_dim);
    // REAL *partial_max,*partial_max_dev;
    // falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    // falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    request_reduction_buffer(n_blocks);
    size_t shared_size = n_threads * sizeof(Real);

    kernel_VecMin<<<grid_dim, block_dim, shared_size>>>(a.devptr, reduction_buffer_device, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(reduction_buffer_host, reduction_buffer_device, sizeof(Real) * n_blocks, MCP::Dev2Hst));
    Real maximum = reduction_buffer_host[0];
    for (Int i = 1; i < n_blocks; i ++) {
        if (reduction_buffer_host[i] < maximum) {
            maximum = reduction_buffer_host[i];
        }
    }

    // falmErrCheckMacro(falmFree(partial_max));
    // falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
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

void FalmMVDevCall::MatrixAdd(Matrix<Real> &a, Matrix<Real> &b, dim3 block_dim) {
    Int n_threads = PRODUCT3(block_dim);
    Int n_blocks = (a.size + n_threads - 1) / n_threads;
    kernel_MatrixAdd<<<n_blocks, n_threads>>>(a.devptr, b.devptr);
    falmWaitStream();
}

__global__ void kernel_Vecaxby(Real a, const MatrixFrame<Real> *vx, Real b, const MatrixFrame<Real> *vy, MatrixFrame<Real> *vresult) {
    const MatrixFrame<Real> &x=*vx, &y=*vy;
    MatrixFrame<Real> &result=*vresult;
    Int tidx  = IDX(threadIdx, blockDim);
    Int bidx  = IDX(blockIdx, gridDim);
    Int bsize = PRODUCT3(blockDim);
    Int gtidx = tidx + bidx * bsize;
    if (gtidx < x.size) {
        result(gtidx) = a*x(gtidx) + b*y(gtidx);
    }
}

void FalmMVDevCall::Vecaxby(Real a, Matrix<Real> &x, Real b, Matrix<Real> &y, Matrix<Real> &result, dim3 block_dim) {
    Int n_threads = PRODUCT3(block_dim);
    Int n_blocks = (x.size + n_threads - 1) / n_threads;
    kernel_Vecaxby<<<n_blocks, n_threads>>>(a, x.devptr, b, y.devptr, result.devptr);
    falmWaitStream();
}

}
