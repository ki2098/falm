#include "../MVL1.h"
#include "devutil.cuh"

namespace Falm {

__global__ void kernel_DotProduct(const MatrixFrame<REAL> *va, const MatrixFrame<REAL> *vb, REAL *partial_sum_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    extern __shared__ REAL cache[];
    const MatrixFrame<REAL> &a=*va, &b=*vb;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    INT tidx = IDX(threadIdx, blockDim);
    REAL tmp = 0;
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
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

REAL L0Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_sum = (REAL*)falmMallocPinned(sizeof(REAL) * n_blocks);
    REAL *partial_sum_dev = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_DotProduct<<<grid_dim, block_dim, shared_size>>>(a.devptr, b.devptr, partial_sum_dev, pdm.shape, map.shape, map.offset);

    falmMemcpy(partial_sum, partial_sum_dev, sizeof(REAL) * n_blocks, MCpType::Dev2Hst);
    REAL sum = partial_sum[0];
    for (INT i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    falmFreePinned(partial_sum);
    falmFreeDevice(partial_sum_dev);

    return sum;
}

__global__ void kernel_EuclideanNormSq(const MatrixFrame<REAL> *va, REAL *partial_sum_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    extern __shared__ REAL cache[];
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    INT tidx = IDX(threadIdx, blockDim);
    REAL tmp = 0;
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdm_shape);
        for (INT __d = 0; __d < a.shape.y; __d ++) {
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

REAL L0Dev_EuclideanNormSq(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_sum = (REAL*)falmMallocPinned(sizeof(REAL) * n_blocks);
    REAL *partial_sum_dev = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_EuclideanNormSq<<<grid_dim, block_dim, shared_size>>>(a.devptr, partial_sum_dev, pdm.shape, map.shape, map.offset);

    falmMemcpy(partial_sum, partial_sum_dev, sizeof(REAL) * n_blocks, MCpType::Dev2Hst);
    REAL sum = partial_sum[0];
    for (INT i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    falmFreePinned(partial_sum);
    falmFreeDevice(partial_sum_dev);

    return sum;
}

/* __global__ void kernel_MaxDiag(const MatrixFrame<REAL> *va, REAL *partial_max_dev, INTx3 pdm_shape, INTx3 map_shape, INTx3 map_offset) {
    extern __shared__ REAL cache[];
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    INT tidx = IDX(threadIdx, blockDim);
    REAL tmp = 0;
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
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
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
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
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
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
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
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
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
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
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max = (REAL*)falmMallocPinned(sizeof(REAL) * n_blocks);
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

REAL L0Dev_MatColMax(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max = (REAL*)falmMallocPinned(sizeof(REAL) * n_blocks);
    REAL *partial_max_dev = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_MatColMax<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, partial_max_dev, pdm.shape, map.shape, map.offset);

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
}

REAL L0Dev_MatColMin(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max = (REAL*)falmMallocPinned(sizeof(REAL) * n_blocks);
    REAL *partial_max_dev = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_MatColMin<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, partial_max_dev, pdm.shape, map.shape, map.offset);

    falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCpType::Dev2Hst);
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] < maximum) {
            maximum = partial_max[i];
        }
    }

    falmFreePinned(partial_max);
    falmFreeDevice(partial_max_dev);

    return maximum;
}

REAL L0Dev_MatColAbsMax(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max = (REAL*)falmMallocPinned(sizeof(REAL) * n_blocks);
    REAL *partial_max_dev = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_MatColAbsMax<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, partial_max_dev, pdm.shape, map.shape, map.offset);

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
}

REAL L0Dev_MatColAbsMin(Matrix<REAL> &a, INT col, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max = (REAL*)falmMallocPinned(sizeof(REAL) * n_blocks);
    REAL *partial_max_dev = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_MatColAbsMin<<<grid_dim, block_dim, shared_size>>>(a.devptr, col, partial_max_dev, pdm.shape, map.shape, map.offset);

    falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCpType::Dev2Hst);
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] < maximum) {
            maximum = partial_max[i];
        }
    }

    falmFreePinned(partial_max);
    falmFreeDevice(partial_max_dev);

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

void L1Dev_ScaleMatrix(Matrix<REAL> &a, REAL scale, dim3 block_dim) {
    INT n_threads = PRODUCT3(block_dim);
    INT n_blocks = (a.size + n_threads - 1) / n_threads;
    kernel_ScaleMatrix<<<n_blocks, n_threads, 0, 0>>>(a.devptr, scale);
    falmWaitStream(0);
}

__global__ void kernel_VecMax(const MatrixFrame<REAL> *va, REAL *partial_max_dev, INT3 pdm_shape, INT3 map_shape, INT3 map_offset ) {
    extern __shared__ REAL cache[];
    const MatrixFrame<REAL> &a=*va;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    INT tidx = IDX(threadIdx, blockDim);
    REAL tmp = 0;
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdm_shape);
        for (INT n = 0; n < a.shape.y; n ++) {
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
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdm_shape);
        for (INT n = 0; n < a.shape.y; n ++) {
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

REAL L0Dev_VecMax(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max = (REAL*)falmMallocPinned(sizeof(REAL) * n_blocks);
    REAL *partial_max_dev = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_VecMax<<<grid_dim, block_dim, shared_size>>>(a.devptr, partial_max_dev, pdm.shape, map.shape, map.offset);

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
}

REAL L0Dev_VecMin(Matrix<REAL> &a, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max = (REAL*)falmMallocPinned(sizeof(REAL) * n_blocks);
    REAL *partial_max_dev = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_VecMin<<<grid_dim, block_dim, shared_size>>>(a.devptr, partial_max_dev, pdm.shape, map.shape, map.offset);

    falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCpType::Dev2Hst);
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] < maximum) {
            maximum = partial_max[i];
        }
    }

    falmFreePinned(partial_max);
    falmFreeDevice(partial_max_dev);

    return maximum;
}

}
