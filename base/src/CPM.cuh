#ifndef FALM_CPM_H
#define FALM_CPM_H

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "Mapper.cuh"
#include "Util.cuh"
#include "param.h"

namespace FALM {

namespace CPM {

struct Request {
    void         *buffer;
    int           buflen;
    int           srcdst;
    int              tag;
    MPI_Request *request;
    MPI_Status   *status;
    MPI_Datatype   dtype;
    MPI_Comm        comm;
    Mapper           map;
    Request() : status(MPI_STATUSES_IGNORE) {};
};

__global__ static void cpm_pack_buffer_kernel(double *src, double *buffer, Mapper domain, Mapper range) {
    unsigned int i, j, k;
    UTIL::THREAD2IJK(i, j, k);
    dim3 &size = domain.size;
    if (i < range.size.x && j < range.size.y && k < range.size.z) {
        unsigned int buffer_idx = UTIL::IDX(i, j, k, range.size);
        i += range.offset.x;
        j += range.offset.y;
        k += range.offset.z;
        unsigned int src_idx = UTIL::IDX(i, j, k, size);
        buffer[buffer_idx] = src[src_idx];
    }
}

static void cpm_pack_buffer(double *src, Mapper &domain, Request &req, unsigned int loc) {
    Mapper &range = req.map;
    if (loc == LOC::DEVICE) {
        double *buffer;
        cudaMalloc(&buffer, sizeof(double) * range.num);
        dim3 block(
            min(range.size.x, block_size.x), 
            min(range.size.y, block_size.y), 
            min(range.size.z, block_size.z)
        ); 
        dim3 grid(
            (range.size.x + block.x - 1) / block.x,
            (range.size.y + block.y - 1) / block.y,
            (range.size.z + block.z - 1) / block.z
        );
        cpm_pack_buffer_kernel<<<grid, block>>>(src, buffer, domain, range);
        req.buffer = buffer;
    }
    req.dtype = MPI_DOUBLE;
    req.buflen = req.map.num;
}

__global__ static void cpm_pack_buffer_colored_kernel(double *src, double *buffer, Mapper domain, Mapper range, unsigned int color) {
    unsigned int i, j, k;
    UTIL::THREAD2IJK(i, j, k);
    dim3 &size = domain.size;
    dim3 &origin = domain.offset;
    if (i < range.size.x && j < range.size.y && k < range.size.z) {
        unsigned int buffer_idx = UTIL::IDX(i, j, k, range.size) / 2;
        i += range.offset.x;
        j += range.offset.y;
        k += range.offset.z;
        unsigned int src_idx = UTIL::IDX(i, j, k, size);
        i += domain.offset.x;
        j += domain.offset.y;
        k += domain.offset.z;
        if ((i + j + k) % 2 == color) {
            buffer[buffer_idx] = src[src_idx];
        }
    }
}

static void cpm_pack_buffer_colored(double *src, Mapper &domain, Request &req, unsigned int color, unsigned int loc) {
    Mapper &range = req.map;
    unsigned int ref_color = (domain.offset.x + domain.offset.y + domain.offset.z + range.offset.x + range.offset.y + range.offset.z) % 2;
    unsigned int color_num = range.num / 2;
    if (ref_color == color && range.num % 2 == 1) {
        color_num ++;
    }
    if (loc == LOC::DEVICE) {
        double *buffer;
        cudaMalloc(&buffer, sizeof(double) * color_num);
        dim3 block(
            min(range.size.x, block_size.x), 
            min(range.size.y, block_size.y), 
            min(range.size.z, block_size.z)
        ); 
        dim3 grid(
            (range.size.x + block.x - 1) / block.x,
            (range.size.y + block.y - 1) / block.y,
            (range.size.z + block.z - 1) / block.z
        );
        cpm_pack_buffer_colored_kernel<<<grid, block>>>(src, buffer, domain, range, color);
        req.buffer = buffer;
    }
    req.dtype = MPI_DOUBLE;
    req.buflen = color_num;
}

__global__ static void cpm_unpack_buffer_kernel(double *dst, double *buffer, Mapper domain, Mapper range) {
    unsigned int i, j, k;
    UTIL::THREAD2IJK(i, j, k);
    dim3 &size = domain.size;
    dim3 &origin = domain.offset;
    if (i < range.size.x && j < range.size.y && k < range.size.z) {
        unsigned int buffer_idx = UTIL::IDX(i, j, k, range.size);
        i += range.offset.x;
        j += range.offset.y;
        k += range.offset.z;
        unsigned int dst_idx = UTIL::IDX(i, j, k, size);
        dst[dst_idx] = buffer[buffer_idx];
    }
}

static void cpm_unpack_buffer(double *dst, Mapper &domain, Request &req, unsigned int loc) {
    Mapper &range = req.map;
    if (loc == LOC::DEVICE) {
        double *buffer = (double*)req.buffer;
        dim3 block(
            min(range.size.x, block_size.x), 
            min(range.size.y, block_size.y), 
            min(range.size.z, block_size.z)
        ); 
        dim3 grid(
            (range.size.x + block.x - 1) / block.x,
            (range.size.y + block.y - 1) / block.y,
            (range.size.z + block.z - 1) / block.z
        );
        cpm_unpack_buffer_kernel<<<grid, block>>>(dst, buffer, domain, range);
        cudaFree(req.buffer);
        req.buffer == nullptr;
    }
}

__global__ static void cpm_unpack_buffer_kernel_colored (double *dst, double *buffer, Mapper domain, Mapper range, unsigned int color) {
    unsigned int i, j, k;
    UTIL::THREAD2IJK(i, j, k);
    dim3 &size = domain.size;
    dim3 &origin = domain.offset;
    if (i < range.size.x && j < range.size.y && k < range.size.z) {
        unsigned int buffer_idx = UTIL::IDX(i, j, k, range.size) / 2;
        i += range.offset.x;
        j += range.offset.y;
        k += range.offset.z;
        unsigned int dst_idx = UTIL::IDX(i, j, k, size);
        i += domain.offset.x;
        j += domain.offset.y;
        k += domain.offset.z;
        if ((i + j + k) % 2 == color) {
            dst[dst_idx] = buffer[buffer_idx];
        }
    }
}

static void cpm_unpack_buffer_colored(double *dst, Mapper &domain, Request &req, unsigned int color, unsigned int loc) {
    Mapper &range = req.map;
    if (loc == LOC::DEVICE) {
        double *buffer = (double*)req.buffer;
        dim3 block(
            min(range.size.x, block_size.x), 
            min(range.size.y, block_size.y), 
            min(range.size.z, block_size.z)
        ); 
        dim3 grid(
            (range.size.x + block.x - 1) / block.x,
            (range.size.y + block.y - 1) / block.y,
            (range.size.z + block.z - 1) / block.z
        );
        cpm_unpack_buffer_kernel_colored<<<grid, block>>>(dst, buffer, domain, range, color);
        cudaFree(req.buffer);
        req.buffer == nullptr;
    }
}

static void cpm_isend(Request &req, int tag, int dst, MPI_Comm comm) {
    MPI_Isend(req.buffer, req.buflen, req.dtype, dst, tag, comm, req.request);
    req.srcdst = dst;
    req.tag = tag;
    req.comm = comm;
}

static void cpm_irecv(Request &req, int tag, int src, MPI_Comm comm) {
    MPI_Irecv(req.buffer, req.buflen, req.dtype, src, tag, comm, req.request);
    req.srcdst = src;
    req.tag = tag;
    req.comm = comm;
}

static void cpm_wait(Request &req) {
    MPI_Wait(req.request, req.status);
}

static void cpm_waitall(Request *req, int n) {
    MPI_Waitall(n, req[0].request, req[0].status);
}

}

}

#endif