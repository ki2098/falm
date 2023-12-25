#include "../MVDevCall.h"
#include "devutil.cuh"

namespace Falm {

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

void FalmMVDevCall::MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Region &pdm, const Region &map, dim3 block_dim, STREAM stream) {
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

__global__ void kernel_DotProduct(const MatrixFrame<REAL> *va, const MatrixFrame<REAL> *vb, REAL *partial_sum_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    extern __shared__ REAL cache[];
    const MatrixFrame<REAL> &a=*va, &b=*vb;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    INT tidx = IDX(threadIdx, blockDim);
    REAL tmp = 0;
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        tmp = a(idx) * b(idx);
    }
    cache[tidx] = tmp;
    __syncthreads();

    INT length = PRODUCT3(blockDim);
    while (length > 1) {
        INT cut = length / 2;
        INT reduce = length - cut;
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

REAL FalmMVDevCall::DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_sum,*partial_sum_dev;
    falmErrCheckMacro(falmMalloc((void**)&partial_sum, sizeof(REAL) * n_blocks));
    falmErrCheckMacro(falmMallocDevice((void**)&partial_sum_dev, sizeof(REAL) * n_blocks));
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_DotProduct<<<grid_dim, block_dim, shared_size>>>(a.devptr, b.devptr, partial_sum_dev, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(partial_sum, partial_sum_dev, sizeof(REAL) * n_blocks, MCP::Dev2Hst));
    REAL sum = partial_sum[0];
    for (INT i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    falmErrCheckMacro(falmFree(partial_sum));
    falmErrCheckMacro(falmFreeDevice(partial_sum_dev));

    return sum;
}

__global__ void kernel_EuclideanNormSq(const MatrixFrame<REAL> *va, REAL *partial_sum_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
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
        for (INT __d = 0; __d < a.shape[1]; __d ++) {
            tmp += a(idx, __d) * a(idx, __d);
        }
    }
    cache[tidx] = tmp;
    __syncthreads();

    INT length = PRODUCT3(blockDim);
    while (length > 1) {
        INT cut = length / 2;
        INT reduce = length - cut;
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

REAL FalmMVDevCall::EuclideanNormSq(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_sum,*partial_sum_dev;
    falmErrCheckMacro(falmMalloc((void**)&partial_sum, sizeof(REAL) * n_blocks));
    falmErrCheckMacro(falmMallocDevice((void**)&partial_sum_dev, sizeof(REAL) * n_blocks));
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_EuclideanNormSq<<<grid_dim, block_dim, shared_size>>>(a.devptr, partial_sum_dev, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(partial_sum, partial_sum_dev, sizeof(REAL) * n_blocks, MCP::Dev2Hst));
    REAL sum = partial_sum[0];
    for (INT i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    falmErrCheckMacro(falmFree(partial_sum));
    falmErrCheckMacro(falmFreeDevice(partial_sum_dev));

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

__global__ void kernel_MatColMax(const MatrixFrame<REAL> *va, INT col, REAL *partial_max_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
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
        tmp = a(idx, col);
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
}

__global__ void kernel_MatColMin(const MatrixFrame<REAL> *va, INT col, REAL *partial_max_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
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
        tmp = a(idx, col);
    }
    cache[tidx] = tmp;
    __syncthreads();

    INT length = PRODUCT3(blockDim);
    while (length > 1) {
        INT cut = length / 2;
        INT reduce = length - cut;
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

__global__ void kernel_MatColAbsMax(const MatrixFrame<REAL> *va, INT col, REAL *partial_max_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
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
        tmp = fabs(a(idx, col));
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
}

__global__ void kernel_MatColAbsMin(const MatrixFrame<REAL> *va, INT col, REAL *partial_max_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
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
        tmp = fabs(a(idx, col));
    }
    cache[tidx] = tmp;
    __syncthreads();

    INT length = PRODUCT3(blockDim);
    while (length > 1) {
        INT cut = length / 2;
        INT reduce = length - cut;
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

REAL FalmMVDevCall::MatColMax(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max,*partial_max_dev;
    falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_MatColMax<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, partial_max_dev, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCP::Dev2Hst));
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] > maximum) {
            maximum = partial_max[i];
        }
    }

    falmErrCheckMacro(falmFree(partial_max));
    falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
}

REAL FalmMVDevCall::MatColMin(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max,*partial_max_dev;
    falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_MatColMin<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, partial_max_dev, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCP::Dev2Hst));
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] < maximum) {
            maximum = partial_max[i];
        }
    }

    falmErrCheckMacro(falmFree(partial_max));
    falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
}

REAL FalmMVDevCall::MatColAbsMax(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max,*partial_max_dev;
    falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_MatColAbsMax<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, partial_max_dev, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCP::Dev2Hst));
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] > maximum) {
            maximum = partial_max[i];
        }
    }

    falmErrCheckMacro(falmFree(partial_max));
    falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
}

REAL FalmMVDevCall::MatColAbsMin(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max,*partial_max_dev;
    falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_MatColAbsMin<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, partial_max_dev, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCP::Dev2Hst));
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] < maximum) {
            maximum = partial_max[i];
        }
    }

    falmErrCheckMacro(falmFree(partial_max));
    falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
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

void FalmMVDevCall::ScaleMatrix(Matrix<REAL> &a, REAL scale, dim3 block_dim) {
    INT n_threads = PRODUCT3(block_dim);
    INT n_blocks = (a.size + n_threads - 1) / n_threads;
    kernel_ScaleMatrix<<<n_blocks, n_threads>>>(a.devptr, scale);
    falmWaitStream();
}

__global__ void kernel_VecMax(const MatrixFrame<REAL> *va, REAL *partial_max_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset ) {
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
        for (INT n = 0; n < a.shape[1]; n ++) {
            tmp += a(idx, n) * a(idx, n);
        }
        tmp = sqrt(tmp);
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
}

__global__ void kernel_VecMin(const MatrixFrame<REAL> *va, REAL *partial_max_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset ) {
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
        for (INT n = 0; n < a.shape[1]; n ++) {
            tmp += a(idx, n) * a(idx, n);
        }
        tmp = sqrt(tmp);
    }
    cache[tidx] = tmp;
    __syncthreads();

    INT length = PRODUCT3(blockDim);
    while (length > 1) {
        INT cut = length / 2;
        INT reduce = length - cut;
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

REAL FalmMVDevCall::VecMax(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max,*partial_max_dev;
    falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_VecMax<<<grid_dim, block_dim, shared_size>>>(a.devptr, partial_max_dev, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCP::Dev2Hst));
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] > maximum) {
            maximum = partial_max[i];
        }
    }

    falmErrCheckMacro(falmFree(partial_max));
    falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
}

REAL FalmMVDevCall::VecMin(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max,*partial_max_dev;
    falmErrCheckMacro(falmMalloc((void**)&partial_max, sizeof(REAL) * n_blocks));
    falmErrCheckMacro(falmMallocDevice((void**)&partial_max_dev, sizeof(REAL) * n_blocks));
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_VecMin<<<grid_dim, block_dim, shared_size>>>(a.devptr, partial_max_dev, pdm.shape, map.shape, map.offset);

    falmErrCheckMacro(falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCP::Dev2Hst));
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] < maximum) {
            maximum = partial_max[i];
        }
    }

    falmErrCheckMacro(falmFree(partial_max));
    falmErrCheckMacro(falmFreeDevice(partial_max_dev));

    return maximum;
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

void FalmMVDevCall::MatrixAdd(Matrix<REAL> &a, Matrix<REAL> &b, dim3 block_dim) {
    INT n_threads = PRODUCT3(block_dim);
    INT n_blocks = (a.size + n_threads - 1) / n_threads;
    kernel_MatrixAdd<<<n_blocks, n_threads>>>(a.devptr, b.devptr);
    falmWaitStream();
}

}
