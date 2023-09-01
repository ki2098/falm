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

const int SEND = 0U;
const int RECV = 1U;

struct Request {
    void         *buffer;
    int           buflen;
    int           srcdst;
    int              tag;
    unsigned int   color;
    unsigned int  bufloc;
    int         sendrecv;
    MPI_Request *request;
    MPI_Datatype   dtype;
    MPI_Comm        comm;
    Mapper           map;
    Request() : buffer(nullptr), bufloc(LOC::NONE) {};
    void release() {
        if (bufloc == LOC::HOST) {
            free(buffer);
        } else if (bufloc == LOC::DEVICE) {
            cudaFree(buffer);
        }
        bufloc = LOC::NONE;
    }
};

static void cpm_alloc_buffer(Request &req, unsigned int loc) {
    assert(req.bufloc == LOC::NONE);
    Mapper &range = req.map;
    if (loc == LOC::DEVICE) {
        double *buffer;
        cudaMalloc(&buffer, sizeof(double) * range.num);
        req.buffer = buffer;
    }
    req.buflen = range.num;
    req.bufloc = loc;
}

static void cpm_alloc_buffer_colored(Request &req, Mapper &domain, unsigned int color, unsigned int loc) {
    assert(req.bufloc == LOC::NONE);
    Mapper &range = req.map;
    unsigned int ref_color = (domain.offset.x + domain.offset.y + domain.offset.z + range.offset.x + range.offset.y + range.offset.z) % 2;
    unsigned int color_num = range.num / 2;
    if (ref_color == color && range.num % 2 == 1) {
        color_num ++;
    }
    if (loc == LOC::DEVICE) {
        double *buffer;
        cudaMalloc(&buffer, sizeof(double) * color_num);
        cudaMemset(buffer, 0, sizeof(double) * color_num);
        req.buffer = buffer;
    }
    req.buflen = color_num;
    req.bufloc = loc;
}

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

__global__ static void cpm_pack_buffer_kernel_colored(double *src, double *buffer, Mapper domain, Mapper range, unsigned int color) {
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

static void cpm_pack_buffer(double *src, Mapper &domain, Request &req, unsigned int loc) {
    Mapper &range = req.map;
    if (loc == LOC::DEVICE) {
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
        cpm_pack_buffer_kernel<<<grid, block>>>(src, (double*)req.buffer, domain, range);
    }
}

static void cpm_pack_buffer_colored(double *src, Mapper &domain, Request &req, unsigned int color, unsigned int loc) {
    Mapper &range = req.map;
    if (loc == LOC::DEVICE) {
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
        cpm_pack_buffer_kernel_colored<<<grid, block>>>(src, (double*)req.buffer, domain, range, color);
    }
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

__global__ static void cpm_unpack_buffer_kernel_colored(double *dst, double *buffer, Mapper domain, Mapper range, unsigned int color) {
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
    }
    req.release();
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
    }
    req.release();
}

static void cpm_isend(double *src, Mapper &domain, Request &req, unsigned int loc, int dst, int tag, MPI_Comm comm) {
    req.srcdst = dst;
    req.tag = tag;
    req.comm = comm;
    req.dtype = MPI_DOUBLE;
    req.sendrecv = SEND;
    cpm_alloc_buffer(req, loc);
    cpm_pack_buffer(src, domain, req, loc);
    MPI_Isend(req.buffer, req.buflen, req.dtype, dst, tag, comm, req.request);
}

static void cpm_isend_colored(double *src, Mapper &domain, Request &req, unsigned int color, unsigned int loc, int dst, int tag, MPI_Comm comm) {
    req.srcdst = dst;
    req.tag = tag;
    req.comm = comm;
    req.dtype = MPI_DOUBLE;
    req.color = color;
    req.sendrecv = SEND;
    cpm_alloc_buffer_colored(req, domain, color, loc);
    cpm_pack_buffer_colored(src, domain, req, color, loc);
    MPI_Isend(req.buffer, req.buflen, req.dtype, dst, tag, comm, req.request);
}

static void cpm_irecv(Mapper &domain, Request &req, unsigned int loc, int src, int tag, MPI_Comm comm) {
    req.srcdst = src;
    req.tag = tag;
    req.comm = comm;
    req.dtype = MPI_DOUBLE;
    req.sendrecv = RECV;
    cpm_alloc_buffer(req, loc);
    MPI_Irecv(req.buffer, req.buflen, req.dtype, src, tag, comm, req.request);
}

static void cpm_irecv_colored(Mapper &domain, Request &req, unsigned int color, unsigned int loc, int src, int tag, MPI_Comm comm) {
    req.srcdst = src;
    req.tag = tag;
    req.comm = comm;
    req.dtype = MPI_DOUBLE;
    req.color = color;
    req.sendrecv = RECV;
    cpm_alloc_buffer_colored(req, domain, color, loc);
    MPI_Irecv(req.buffer, req.buflen, req.dtype, src, tag, comm, req.request);
}

static void cpm_wait(Request &req, MPI_Status *status) {
    MPI_Wait(req.request, status);
    if (req.sendrecv == SEND) {
        req.release();
    }
}

static void cpm_waitall(Request *req, int n, MPI_Status *status) {
    MPI_Waitall(n, req[0].request, status);
    for (int i = 0; i < n; i ++) {
        if (req[i].sendrecv == SEND) {
            req[i].release();
        }
    }
}

}

}

#endif