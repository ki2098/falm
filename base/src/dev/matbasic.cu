#include "../matbasic.h"
#include "devutil.cuh"
#include "devparam.cuh"

namespace Falm {

__global__ void calc_dot_product_kernel(MatrixFrame<double> &a, MatrixFrame<double> &b, double *partial_sum_dev, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
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

double dev_calc_dot_product(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    unsigned int n_blocks = PRODUCT3(grid_dim);
    unsigned int n_threads = PRODUCT3(block_dim);
    double *partial_sum = (double*)falmHostMalloc(sizeof(double) * n_blocks);
    double *partial_sum_dev = (double*)falmDevMalloc(sizeof(double) * n_blocks);

    calc_dot_product_kernel<<<grid_dim, block_dim, n_threads * sizeof(double)>>>(*(a.devptr), *(b.devptr), partial_sum_dev, pdom.shape, map.shape, map.offset);

    falmMemcpy(partial_sum, partial_sum_dev, sizeof(double) * n_blocks, MCPTYPE::Dev2Hst);
    double sum = partial_sum[0];
    for (int i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    falmHostFreePtr(partial_sum);
    falmDevFreePtr(partial_sum_dev);

    return sum;
}

__global__ void calc_norm2_sq_kernel(MatrixFrame<double> &a, double *partial_sum_dev, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
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

double dev_calc_norm2_sq(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    unsigned int n_blocks = PRODUCT3(grid_dim);
    unsigned int n_threads = PRODUCT3(block_dim);
    double *partial_sum = (double*)falmHostMalloc(sizeof(double) * n_blocks);
    double *partial_sum_dev = (double*)falmDevMalloc(sizeof(double) * n_blocks);

    calc_norm2_sq_kernel<<<grid_dim, block_dim, n_threads * sizeof(double)>>>(*(a.devptr), partial_sum_dev, pdom.shape, map.shape, map.offset);

    falmMemcpy(partial_sum, partial_sum_dev, sizeof(double) * n_blocks, MCPTYPE::Dev2Hst);
    double sum = partial_sum[0];
    for (int i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    falmHostFreePtr(partial_sum);
    falmDevFreePtr(partial_sum_dev);

    return sum;
}

}