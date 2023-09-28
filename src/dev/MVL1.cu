#include "../MVL1.h"
#include "devutil.cuh"

namespace Falm {

__global__ void kernel_DotProduct(MatrixFrame<REAL> &a, MatrixFrame<REAL> &b, REAL *partial_sum_dev, INTx3 pdom_shape, INTx3 map_shape, INTx3 map_offset) {
    extern __shared__ REAL cache[];
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    INT tidx = IDX(threadIdx, blockDim);
    REAL tmp = 0;
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdom_shape);
        tmp = a(idx) * b(idx);
    }
    cache[tidx] = tmp;

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

REAL L0Dev_DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_sum = (REAL*)falmHostMalloc(sizeof(REAL) * n_blocks);
    REAL *partial_sum_dev = (REAL*)falmDevMalloc(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_DotProduct<<<grid_dim, block_dim, shared_size, cudaStreamPerThread>>>(*(a.devptr), *(b.devptr), partial_sum_dev, pdom.shape, map.shape, map.offset);

    falmMemcpy(partial_sum, partial_sum_dev, sizeof(REAL) * n_blocks, MCpType::Dev2Hst);
    REAL sum = partial_sum[0];
    for (INT i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    falmHostFreePtr(partial_sum);
    falmDevFreePtr(partial_sum_dev);

    return sum;
}

__global__ void kernel_Norm2Sq(MatrixFrame<REAL> &a, REAL *partial_sum_dev, INTx3 pdom_shape, INTx3 map_shape, INTx3 map_offset) {
    extern __shared__ REAL cache[];
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    INT tidx = IDX(threadIdx, blockDim);
    REAL tmp = 0;
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdom_shape);
        tmp = a(idx) * a(idx);
    }
    cache[tidx] = tmp;

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

REAL L0Dev_Norm2Sq(Matrix<REAL> &a, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_sum = (REAL*)falmHostMalloc(sizeof(REAL) * n_blocks);
    REAL *partial_sum_dev = (REAL*)falmDevMalloc(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_Norm2Sq<<<grid_dim, block_dim, shared_size, cudaStreamPerThread>>>(*(a.devptr), partial_sum_dev, pdom.shape, map.shape, map.offset);

    falmMemcpy(partial_sum, partial_sum_dev, sizeof(REAL) * n_blocks, MCpType::Dev2Hst);
    REAL sum = partial_sum[0];
    for (INT i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    falmHostFreePtr(partial_sum);
    falmDevFreePtr(partial_sum_dev);

    return sum;
}

__global__ void kernel_MaxDiag(MatrixFrame<REAL> &a, REAL *partial_max_dev, INTx3 pdom_shape, INTx3 map_shape, INTx3 map_offset) {
    extern __shared__ REAL cache[];
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    INT tidx = IDX(threadIdx, blockDim);
    REAL tmp = 0;
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdom_shape);
        tmp = fabs(a(idx));
    }
    cache[tidx] = tmp;

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

REAL L0Dev_MaxDiag(Matrix<REAL> &a, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_max = (REAL*)falmHostMalloc(sizeof(REAL) * n_blocks);
    REAL *partial_max_dev = (REAL*)falmDevMalloc(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    kernel_MaxDiag<<<grid_dim, block_dim, shared_size, cudaStreamPerThread>>>(*(a.devptr), partial_max_dev, pdom.shape, map.shape, map.offset);

    falmMemcpy(partial_max, partial_max_dev, sizeof(REAL) * n_blocks, MCpType::Dev2Hst);
    REAL maximum = partial_max[0];
    for (INT i = 1; i < n_blocks; i ++) {
        if (partial_max[i] > maximum) {
            maximum = partial_max[i];
        }
    }

    falmHostFreePtr(partial_max);
    falmDevFreePtr(partial_max_dev);

    return maximum;
}

__global__ void kernel_ScaleMatrix(MatrixFrame<REAL> &a, REAL scale) {
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
    kernel_ScaleMatrix<<<n_blocks, n_threads, 0, cudaStreamPerThread>>>(*(a.devptr), scale);
}

}
