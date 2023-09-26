#include "../MVL1.h"
#include "devutil.cuh"

namespace Falm {

__global__ void kernel_DotProduct(MatrixFrame<double> &a, MatrixFrame<double> &b, double *partial_sum_dev, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
    extern __shared__ double cache[];
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    unsigned int tidx = IDX(threadIdx, blockDim);
    double tmp = 0;
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idx = IDX(i, j, k, pdom_shape);
        tmp = a(idx) * b(idx);
    }
    cache[tidx] = tmp;

    unsigned int length = PRODUCT3(blockDim);
    while (length > 1) {
        unsigned int cut = length / 2;
        unsigned int reduce = length - cut;
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

double L0Dev_DotProduct(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    unsigned int n_blocks = PRODUCT3(grid_dim);
    unsigned int n_threads = PRODUCT3(block_dim);
    double *partial_sum = (double*)falmHostMalloc(sizeof(double) * n_blocks);
    double *partial_sum_dev = (double*)falmDevMalloc(sizeof(double) * n_blocks);
    size_t shared_size = n_threads * sizeof(double);

    kernel_DotProduct<<<grid_dim, block_dim, shared_size, 0>>>(*(a.devptr), *(b.devptr), partial_sum_dev, pdom.shape, map.shape, map.offset);

    falmMemcpy(partial_sum, partial_sum_dev, sizeof(double) * n_blocks, MCpType::Dev2Hst);
    double sum = partial_sum[0];
    for (int i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    falmHostFreePtr(partial_sum);
    falmDevFreePtr(partial_sum_dev);

    return sum;
}

__global__ void kernel_Norm2Sq(MatrixFrame<double> &a, double *partial_sum_dev, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
    extern __shared__ double cache[];
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    unsigned int tidx = IDX(threadIdx, blockDim);
    double tmp = 0;
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idx = IDX(i, j, k, pdom_shape);
        tmp = a(idx) * a(idx);
    }
    cache[tidx] = tmp;

    unsigned int length = PRODUCT3(blockDim);
    while (length > 1) {
        unsigned int cut = length / 2;
        unsigned int reduce = length - cut;
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

double L0Dev_Norm2Sq(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    unsigned int n_blocks = PRODUCT3(grid_dim);
    unsigned int n_threads = PRODUCT3(block_dim);
    double *partial_sum = (double*)falmHostMalloc(sizeof(double) * n_blocks);
    double *partial_sum_dev = (double*)falmDevMalloc(sizeof(double) * n_blocks);
    size_t shared_size = n_threads * sizeof(double);

    kernel_Norm2Sq<<<grid_dim, block_dim, shared_size, 0>>>(*(a.devptr), partial_sum_dev, pdom.shape, map.shape, map.offset);

    falmMemcpy(partial_sum, partial_sum_dev, sizeof(double) * n_blocks, MCpType::Dev2Hst);
    double sum = partial_sum[0];
    for (int i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    falmHostFreePtr(partial_sum);
    falmDevFreePtr(partial_sum_dev);

    return sum;
}

__global__ void kernel_MaxDiag(MatrixFrame<double> &a, double *partial_max_dev, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
    extern __shared__ double cache[];
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    unsigned int tidx = IDX(threadIdx, blockDim);
    double tmp = 0;
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idx = IDX(i, j, k, pdom_shape);
        tmp = fabs(a(idx));
    }
    cache[tidx] = tmp;

    unsigned int length = PRODUCT3(blockDim);
    while (length > 1) {
        unsigned int cut = length / 2;
        unsigned int reduce = length - cut;
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

double L0Dev_MaxDiag(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    unsigned int n_blocks = PRODUCT3(grid_dim);
    unsigned int n_threads = PRODUCT3(block_dim);
    double *partial_max = (double*)falmHostMalloc(sizeof(double) * n_blocks);
    double *partial_max_dev = (double*)falmDevMalloc(sizeof(double) * n_blocks);
    size_t shared_size = n_threads * sizeof(double);

    kernel_MaxDiag<<<grid_dim, block_dim, shared_size, 0>>>(*(a.devptr), partial_max_dev, pdom.shape, map.shape, map.offset);

    falmMemcpy(partial_max, partial_max_dev, sizeof(double) * n_blocks, MCpType::Dev2Hst);
    double maximum = partial_max[0];
    for (int i = 1; i < n_blocks; i ++) {
        if (partial_max[i] > maximum) {
            maximum = partial_max[i];
        }
    }

    falmHostFreePtr(partial_max);
    falmDevFreePtr(partial_max_dev);

    return maximum;
}

__global__ void kernel_ScaleMatrix(MatrixFrame<double> &a, double scale) {
    unsigned int tidx  = IDX(threadIdx, blockDim);
    unsigned int bidx  = IDX(blockIdx, gridDim);
    unsigned int bsize = PRODUCT3(blockDim);
    unsigned int gtidx = tidx + bidx * bsize;
    if (gtidx < a.size) {
        a(gtidx) /= scale;
    }
}

void L1Dev_ScaleMatrix(Matrix<double> &a, double scale, dim3 block_dim) {
    unsigned int n_threads = PRODUCT3(block_dim);
    unsigned int n_blocks = (a.size + n_threads - 1) / n_threads;
    kernel_ScaleMatrix<<<n_blocks, n_threads, 0, 0>>>(*(a.devptr), scale);
}

}
