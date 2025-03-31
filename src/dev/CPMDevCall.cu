#include "../typedef.h"
#include "../CPMDevCall.h"
#include "devutil.cuh"

namespace Falm {

__global__ void kernel_CPM_PackBuffer(Real *buffer, Int3 buf_shape, Int3 buf_offset, Real *src, Int3 src_shape) {
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape[0] && j < buf_shape[1] && k < buf_shape[2]) {
        Int buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset[0];
        j += buf_offset[1];
        k += buf_offset[2];
        Int src_idx = IDX(i, j, k, src_shape);
        buffer[buf_idx] = src[src_idx];
    }
}

__global__ void kernel_CPM_PackColoredBuffer(Real *buffer, Int3 buf_shape, Int3 buf_offset, Int color, Real *src, Int3 src_shape, Int3 src_offset) {
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape[0] && j < buf_shape[1] && k < buf_shape[2]) {
        Int buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset[0];
        j += buf_offset[1];
        k += buf_offset[2];
        Int src_idx = IDX(i, j, k, src_shape);
        if ((i + j + k + SUM3(src_offset)) % 2 == color) {
            buffer[buf_idx / 2] = src[src_idx];
        }
    }
}

__global__ void kernel_CPM_UnpackBuffer(Real *buffer, Int3 buf_shape, Int3 buf_offset, Real *dst, Int3 dst_shape) {
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape[0] && j < buf_shape[1] && k < buf_shape[2]) {
        Int buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset[0];
        j += buf_offset[1];
        k += buf_offset[2];
        Int dst_idx = IDX(i, j, k, dst_shape);
        dst[dst_idx] = buffer[buf_idx];
    }
}

__global__ void kernel_CPM_UnpackColoredBuffer(Real *buffer, Int3 buf_shape, Int3 buf_offset, Int color , Real *dst, Int3 dst_shape, Int3 dst_offset) {
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < buf_shape[0] && j < buf_shape[1] && k < buf_shape[2]) {
        Int buf_idx = IDX(i, j, k, buf_shape);
        i += buf_offset[0];
        j += buf_offset[1];
        k += buf_offset[2];
        Int dst_idx = IDX(i, j, k, dst_shape);
        if ((i + j + k + SUM3(dst_offset)) % 2 == color) {
            dst[dst_idx] = buffer[buf_idx / 2];
        }
    }
}

void CPMDevCall::PackBuffer(Real *buffer, Region &map, Real *src, Region &pdm, dim3 block_dim, Stream stream) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_CPM_PackBuffer<<<grid_dim, block_dim, 0, stream>>>(buffer, map.shape, map.offset, src, pdm.shape);
}

void CPMDevCall::PackColoredBuffer(Real *buffer, Region &map, Int color, Real *src, Region &pdm, dim3 block_dim, Stream stream) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_CPM_PackColoredBuffer<<<grid_dim, block_dim, 0, stream>>>(buffer, map.shape, map.offset, color, src, pdm.shape, pdm.offset);
}

void CPMDevCall::UnpackBuffer(Real *buffer, Region &map, Real *dst, Region &pdm, dim3 block_dim, Stream stream) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_CPM_UnpackBuffer<<<grid_dim, block_dim, 0, stream>>>(buffer, map.shape, map.offset, dst, pdm.shape);
}

void CPMDevCall::UnpackColoredBuffer(Real *buffer, Region &map, Int color, Real *dst, Region &pdm, dim3 block_dim, Stream stream) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_CPM_UnpackColoredBuffer<<<grid_dim, block_dim, 0, stream>>>(buffer, map.shape, map.offset, color, dst, pdm.shape, pdm.offset);
}

// void CPML1Dev_PackBuffer(CPMBuffer &buffer, REAL *src, Region &pdm, dim3 block_dim) {
//     Region &map = buffer.map;
//     dim3 grid_dim(
//         (map.shape[0] + block_dim.x - 1) / block_dim.x,
//         (map.shape[1] + block_dim.y - 1) / block_dim.y,
//         (map.shape[2] + block_dim.z - 1) / block_dim.z
//     );
//     if (buffer.hdctype == HDCType::Device) {
//         kernel_CPM_PackBuffer<<<grid_dim, block_dim, 0, 0>>>((REAL*)buffer.ptr, buffer.map.shape, buffer.map.offset, src, pdm.shape);
//     } else if (buffer.hdctype == HDCType::Host) {
//         REAL *ptr = (REAL*)falmMallocDevice(sizeof(REAL) * buffer.count);
//         kernel_CPM_PackBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, src, pdm.shape);
//         falmMemcpy(buffer.ptr, ptr, sizeof(REAL) * buffer.count, MCpType::Dev2Hst);
//         falmFreeDevice(ptr);
//     }
// }

// void CPML1Dev_PackColoredBuffer(CPMBuffer &buffer, REAL *src, Region &pdm, dim3 block_dim) {
//     Region &map = buffer.map;
//     dim3 grid_dim(
//         (map.shape[0] + block_dim.x - 1) / block_dim.x,
//         (map.shape[1] + block_dim.y - 1) / block_dim.y,
//         (map.shape[2] + block_dim.z - 1) / block_dim.z
//     );
//     if (buffer.hdctype == HDCType::Device) {
//         kernel_CPM_PackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>((REAL*)buffer.ptr, buffer.map.shape, buffer.map.offset, buffer.color, src, pdm.shape, pdm.offset);
//     } else if (buffer.hdctype == HDCType::Host) {
//         REAL *ptr = (REAL*)falmMallocDevice(sizeof(REAL) * buffer.count);
//         kernel_CPM_PackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, buffer.color, src, pdm.shape, pdm.offset);
//         falmMemcpy(buffer.ptr, ptr, sizeof(REAL) * buffer.count, MCpType::Dev2Hst);
//         falmFreeDevice(ptr);
//     }
// }

// void CPML1Dev_UnpackBuffer(CPMBuffer &buffer, REAL *dst, Region &pdm, dim3 block_dim) {
//     Region &map = buffer.map;
//     dim3 grid_dim(
//         (map.shape[0] + block_dim.x - 1) / block_dim.x,
//         (map.shape[1] + block_dim.y - 1) / block_dim.y,
//         (map.shape[2] + block_dim.z - 1) / block_dim.z
//     );
//     if (buffer.hdctype == HDCType::Device) {
//         kernel_CPM_UnpackBuffer<<<grid_dim, block_dim, 0, 0>>>((REAL*)buffer.ptr, buffer.map.shape, buffer.map.offset, dst, pdm.shape);
//     } else if (buffer.hdctype == HDCType::Host) {
//         REAL *ptr = (REAL*)falmMallocDevice(sizeof(REAL) * buffer.count);
//         falmMemcpy(ptr, buffer.ptr, sizeof(REAL) * buffer.count, MCpType::Hst2Dev);
//         kernel_CPM_UnpackBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, dst, pdm.shape);
//         falmFreeDevice(ptr);
//     }
// }

// void CPML1Dev_UnpackColoredBuffer(CPMBuffer &buffer, REAL *dst, Region &pdm, dim3 block_dim) {
//     Region &map = buffer.map;
//     dim3 grid_dim(
//         (map.shape[0] + block_dim.x - 1) / block_dim.x,
//         (map.shape[1] + block_dim.y - 1) / block_dim.y,
//         (map.shape[2] + block_dim.z - 1) / block_dim.z
//     );
//     if (buffer.hdctype == HDCType::Device) {
//         kernel_CPM_UnpackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>((REAL*)buffer.ptr, buffer.map.shape, buffer.map.offset, buffer.color, dst, pdm.shape, pdm.offset);
//     } else if (buffer.hdctype == HDCType::Host) {
//         REAL *ptr = (REAL*)falmMallocDevice(sizeof(REAL) * buffer.count);
//         falmMemcpy(ptr, buffer.ptr, sizeof(REAL) * buffer.count, MCpType::Hst2Dev);
//         kernel_CPM_UnpackColoredBuffer<<<grid_dim, block_dim, 0, 0>>>(ptr, buffer.map.shape, buffer.map.offset, buffer.color, dst, pdm.shape, pdm.offset);
//         falmFreeDevice(ptr);
//     }
// }

}
