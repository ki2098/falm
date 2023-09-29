#include "../typedef.h"
#include "../CPML1v2.h"
#include "devutil.cuh"

namespace Falm {

__global__ void kernel_CPM_PackBuffer(REAL *buffer, INTx3 buf_shape, INTx3 buf_offset, REAL *src, INTx3 src_shape) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape.x && j < buf_shape.y && k < buf_shape.z) {
        INT buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset.x;
        j += buf_offset.y;
        k += buf_offset.z;
        INT src_idx = IDX(i, j, k, src_shape);
        buffer[buf_idx] = src[src_idx];
    }
}

__global__ void kernel_CPM_PackColoredBuffer(REAL *buffer, INTx3 buf_shape, INTx3 buf_offset, INT color, REAL *src, INTx3 src_shape, INTx3 src_offset) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape.x && j < buf_shape.y && k < buf_shape.z) {
        INT buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset.x;
        j += buf_offset.y;
        k += buf_offset.z;
        INT src_idx = IDX(i, j, k, src_shape);
        if ((i + j + k + SUM3(src_offset)) % 2 == color) {
            buffer[buf_idx / 2] = src[src_idx];
        }
    }
}

__global__ void kernel_CPM_UnpackBuffer(REAL *buffer, INTx3 buf_shape, INTx3 buf_offset, REAL *dst, INTx3 dst_shape) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape.x && j < buf_shape.y && k < buf_shape.z) {
        INT buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset.x;
        j += buf_offset.y;
        k += buf_offset.z;
        INT dst_idx = IDX(i, j, k, dst_shape);
        dst[dst_idx] = buffer[buf_idx];
    }
}

__global__ void kernel_CPM_UnpackColoredBuffer(REAL *buffer, INTx3 buf_shape, INTx3 buf_offset, INT color , REAL *dst, INTx3 dst_shape, INTx3 dst_offset) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape.x && j < buf_shape.y && k < buf_shape.z) {
        INT buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset.x;
        j += buf_offset.y;
        k += buf_offset.z;
        INT dst_idx = IDX(i, j, k, dst_shape);
        if ((i + j + k + SUM3(dst_offset)) % 2 == color) {
            dst[dst_idx] = buffer[buf_idx / 2];
        }
    }
}

void CPML0Dev_PackBuffer(REAL *buffer, Mapper &map, REAL *src, Mapper &pdm, dim3 block_dim, STREAM stream) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_CPM_PackBuffer<<<grid_dim, block_dim, 0, stream>>>(buffer, map.shape, map.offset, src, pdm.shape);
}

void CPML0Dev_PackColoredBuffer(REAL *buffer, Mapper &map, INT color, REAL *src, Mapper &pdm, dim3 block_dim, STREAM stream) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_CPM_PackColoredBuffer<<<grid_dim, block_dim, 0, stream>>>(buffer, map.shape, map.offset, color, src, pdm.shape, pdm.offset);
}

void CPML0Dev_UnpackBuffer(REAL *buffer, Mapper &map, REAL *dst, Mapper &pdm, dim3 block_dim, STREAM stream) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_CPM_UnpackBuffer<<<grid_dim, block_dim, 0, stream>>>(buffer, map.shape, map.offset, dst, pdm.shape);
}

void CPML0Dev_UnpackColoredBuffer(REAL *buffer, Mapper &map, INT color, REAL *dst, Mapper &pdm, dim3 block_dim, STREAM stream) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_CPM_UnpackColoredBuffer<<<grid_dim, block_dim, 0, stream>>>(buffer, map.shape, map.offset, color, dst, pdm.shape, pdm.offset);
}

void CPML1Dev_PackBuffer(CPMBuffer &buffer, REAL *src, Mapper &pdm, dim3 block_dim) {
    Mapper &map = buffer.map;
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    if (buffer.hdctype == HDCType::Device) {
        kernel_CPM_PackBuffer<<<grid_dim, block_dim, 0, 0>>>((REAL*)buffer.ptr, buffer.map.shape, buffer.map.offset, src, pdm.shape);
    } else if (buffer.hdctype == HDCType::Host) {
        REAL *ptr = (REAL*)falmMallocDevice(sizeof(REAL) * buffer.count);
        kernel_CPM_PackBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, src, pdm.shape);
        falmMemcpy(buffer.ptr, ptr, sizeof(REAL) * buffer.count, MCpType::Dev2Hst);
        falmFreeDevice(ptr);
    }
}

void CPML1Dev_PackColoredBuffer(CPMBuffer &buffer, REAL *src, Mapper &pdm, dim3 block_dim) {
    Mapper &map = buffer.map;
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    if (buffer.hdctype == HDCType::Device) {
        kernel_CPM_PackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>((REAL*)buffer.ptr, buffer.map.shape, buffer.map.offset, buffer.color, src, pdm.shape, pdm.offset);
    } else if (buffer.hdctype == HDCType::Host) {
        REAL *ptr = (REAL*)falmMallocDevice(sizeof(REAL) * buffer.count);
        kernel_CPM_PackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, buffer.color, src, pdm.shape, pdm.offset);
        falmMemcpy(buffer.ptr, ptr, sizeof(REAL) * buffer.count, MCpType::Dev2Hst);
        falmFreeDevice(ptr);
    }
}

void CPML1Dev_UnpackBuffer(CPMBuffer &buffer, REAL *dst, Mapper &pdm, dim3 block_dim) {
    Mapper &map = buffer.map;
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    if (buffer.hdctype == HDCType::Device) {
        kernel_CPM_UnpackBuffer<<<grid_dim, block_dim, 0, 0>>>((REAL*)buffer.ptr, buffer.map.shape, buffer.map.offset, dst, pdm.shape);
    } else if (buffer.hdctype == HDCType::Host) {
        REAL *ptr = (REAL*)falmMallocDevice(sizeof(REAL) * buffer.count);
        falmMemcpy(ptr, buffer.ptr, sizeof(REAL) * buffer.count, MCpType::Hst2Dev);
        kernel_CPM_UnpackBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, dst, pdm.shape);
        falmFreeDevice(ptr);
    }
}

void CPML1Dev_UnpackColoredBuffer(CPMBuffer &buffer, REAL *dst, Mapper &pdm, dim3 block_dim) {
    Mapper &map = buffer.map;
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    if (buffer.hdctype == HDCType::Device) {
        kernel_CPM_UnpackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>((REAL*)buffer.ptr, buffer.map.shape, buffer.map.offset, buffer.color, dst, pdm.shape, pdm.offset);
    } else if (buffer.hdctype == HDCType::Host) {
        REAL *ptr = (REAL*)falmMallocDevice(sizeof(REAL) * buffer.count);
        falmMemcpy(ptr, buffer.ptr, sizeof(REAL) * buffer.count, MCpType::Hst2Dev);
        kernel_CPM_UnpackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, buffer.color, dst, pdm.shape, pdm.offset);
        falmFreeDevice(ptr);
    }
}

}
