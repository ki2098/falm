#include "../typedef.h"
#include "../CPML1v2.h"
#include "devutil.cuh"

namespace Falm {

__global__ void kernel_CPM_PackBuffer(double *buffer, uint3 buf_shape, uint3 buf_offset, double *src, uint3 src_shape) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape.x && j < buf_shape.y && k < buf_shape.z) {
        unsigned int buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset.x;
        j += buf_offset.y;
        k += buf_offset.z;
        unsigned int src_idx = IDX(i, j, k, src_shape);
        buffer[buf_idx] = src[src_idx];
    }
}

__global__ void kernel_CPM_PackColoredBuffer(double *buffer, uint3 buf_shape, uint3 buf_offset, unsigned int color, double *src, uint3 src_shape, uint3 src_offset) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape.x && j < buf_shape.y && k < buf_shape.z) {
        unsigned int buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset.x;
        j += buf_offset.y;
        k += buf_offset.z;
        unsigned int src_idx = IDX(i, j, k, src_shape);
        if ((i + j + k + SUM3(src_offset)) % 2 == color) {
            buffer[buf_idx / 2] = src[src_idx];
        }
    }
}

__global__ void kernel_CPM_UnpackBuffer(double *buffer, uint3 buf_shape, uint3 buf_offset, double *dst, uint3 dst_shape) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape.x && j < buf_shape.y && k < buf_shape.z) {
        unsigned int buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset.x;
        j += buf_offset.y;
        k += buf_offset.z;
        unsigned int dst_idx = IDX(i, j, k, dst_shape);
        dst[dst_idx] = buffer[buf_idx];
    }
}

__global__ void kernel_CPM_UnpackColoredBuffer(double *buffer, uint3 buf_shape, uint3 buf_offset, unsigned int color , double *dst, uint3 dst_shape, uint3 dst_offset) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape.x && j < buf_shape.y && k < buf_shape.z) {
        unsigned int buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset.x;
        j += buf_offset.y;
        k += buf_offset.z;
        unsigned int dst_idx = IDX(i, j, k, dst_shape);
        if ((i + j + k + SUM3(dst_offset)) % 2 == color) {
            dst[dst_idx] = buffer[buf_idx / 2];
        }
    }
}

void CPML1Dev_PackBuffer(CPMBuffer &buffer, double *src, Mapper &pdom, dim3 block_dim) {
    Mapper &map = buffer.map;
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    if (buffer.hdctype == HDCType::Device) {
        kernel_CPM_PackBuffer<<<grid_dim, block_dim, 0, 0>>>((double*)buffer.ptr, buffer.map.shape, buffer.map.offset, src, pdom.shape);
    } else if (buffer.hdctype == HDCType::Host) {
        double *ptr = (double*)falmDevMalloc(sizeof(double) * buffer.count);
        kernel_CPM_PackBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, src, pdom.shape);
        falmMemcpy(buffer.ptr, ptr, sizeof(double) * buffer.count, MCpType::Dev2Hst);
        falmDevFreePtr(ptr);
    }
}

void CPML1Dev_PackColoredBuffer(CPMBuffer &buffer, double *src, Mapper &pdom, dim3 block_dim) {
    Mapper &map = buffer.map;
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    if (buffer.hdctype == HDCType::Device) {
        kernel_CPM_PackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>((double*)buffer.ptr, buffer.map.shape, buffer.map.offset, buffer.color, src, pdom.shape, pdom.offset);
    } else if (buffer.hdctype == HDCType::Host) {
        double *ptr = (double*)falmDevMalloc(sizeof(double) * buffer.count);
        kernel_CPM_PackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, buffer.color, src, pdom.shape, pdom.offset);
        falmMemcpy(buffer.ptr, ptr, sizeof(double) * buffer.count, MCpType::Dev2Hst);
        falmDevFreePtr(ptr);
    }
}

void CPML1Dev_UnpackBuffer(CPMBuffer &buffer, double *dst, Mapper &pdom, dim3 block_dim) {
    Mapper &map = buffer.map;
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    if (buffer.hdctype == HDCType::Device) {
        kernel_CPM_UnpackBuffer<<<grid_dim, block_dim, 0, 0>>>((double*)buffer.ptr, buffer.map.shape, buffer.map.offset, dst, pdom.shape);
    } else if (buffer.hdctype == HDCType::Host) {
        double *ptr = (double*)falmDevMalloc(sizeof(double) * buffer.count);
        falmMemcpy(ptr, buffer.ptr, sizeof(double) * buffer.count, MCpType::Hst2Dev);
        kernel_CPM_UnpackBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, dst, pdom.shape);
        falmDevFreePtr(ptr);
    }
}

void CPML1Dev_UnpackColoredBuffer(CPMBuffer &buffer, double *dst, Mapper &pdom, dim3 block_dim) {
    Mapper &map = buffer.map;
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    if (buffer.hdctype == HDCType::Device) {
        kernel_CPM_UnpackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>((double*)buffer.ptr, buffer.map.shape, buffer.map.offset, buffer.color, dst, pdom.shape, pdom.offset);
    } else if (buffer.hdctype == HDCType::Host) {
        double *ptr = (double*)falmDevMalloc(sizeof(double) * buffer.count);
        falmMemcpy(ptr, buffer.ptr, sizeof(double) * buffer.count, MCpType::Hst2Dev);
        kernel_CPM_UnpackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, buffer.color, dst, pdom.shape, pdom.offset);
        falmDevFreePtr(ptr);
    }
}

}
