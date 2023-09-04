#ifndef FALM_FIELD_CUH
#define FALM_FIELD_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <mpi.h>
#include "Util.cuh"
#include "Mapper.cuh"
#include "param.h"

namespace FALM {

template<class T>
struct FieldFrame {
    T             *ptr;
    unsigned int  size;
    uint2 shape;
    unsigned int   dim;
    unsigned int   num;
    unsigned int   loc;
    unsigned int label;
    FieldFrame();
    FieldFrame(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel);
    FieldFrame(unsigned int vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel);
    ~FieldFrame();
    void init(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel);
    void init(unsigned int vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel);
    void release();
    __host__ __device__ T& operator()(int idx) {return ptr[idx];}
    __host__ __device__ T& operator()(int idx, int dim_idx) {return ptr[dim_idx * size + idx];}
};

template<class T>
FieldFrame<T>::FieldFrame() : ptr(nullptr), size(0), shape(make_uint2(0,0)), dim(0), num(0), loc(LOC::NONE), label(0) {}

template<class T>
FieldFrame<T>::FieldFrame(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel) : size(vsize.x * vsize.y * vsize.z), dim(vdim), num(vsize.x * vsize.y * vsize.z * vdim), loc(vloc), label(vlabel) {
    if (loc == LOC::HOST) {
        ptr = (T*)malloc(sizeof(T) * num);
        memset(ptr, 0, sizeof(T) * num);
    } else if (loc == LOC::DEVICE) {
        cudaMalloc(&ptr, sizeof(T) * num);
        cudaMemset(ptr, 0, sizeof(T) * num);
    }
}

template<class T>
FieldFrame<T>::FieldFrame(unsigned int vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel) : size(vsize), dim(vdim), num(vsize * vdim), loc(vloc), label(vlabel) {
    if (loc == LOC::HOST) {
        ptr = (T*)malloc(sizeof(T) * num);
        memset(ptr, 0, sizeof(T) * num);
    } else if (loc == LOC::DEVICE) {
        cudaMalloc(&ptr, sizeof(T) * num);
        cudaMemset(ptr, 0, sizeof(T) * num);
    }
}

template<class T>
FieldFrame<T>::~FieldFrame() {
    if (loc == LOC::HOST) {
        free(ptr);
    } else if (loc == LOC::DEVICE) {
        cudaFree(ptr);
    }
    loc = LOC::NONE;
}

template<class T>
void FieldFrame<T>::init(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel) {
    assert(loc == LOC::NONE);
    size  = vsize.x * vsize.y * vsize.z;
    dim   = vdim;
    num   = size * dim;
    loc   = vloc;
    label = vlabel;
    if (loc == LOC::HOST) {
        ptr = (T*)malloc(sizeof(T) * num);
        memset(ptr, 0, sizeof(T) * num);
    } else if (loc == LOC::DEVICE) {
        cudaMalloc(&ptr, sizeof(T) * num);
        cudaMemset(ptr, 0, sizeof(T) * num);
    }
}

template<class T>
void FieldFrame<T>::init(unsigned int vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel) {
    assert(loc == LOC::NONE);
    size  = vsize;
    dim   = vdim;
    num   = size * dim;
    loc   = vloc;
    label = vlabel;
    if (loc == LOC::HOST) {
        ptr = (T*)malloc(sizeof(T) * num);
        memset(ptr, 0, sizeof(T) * num);
    } else if (loc == LOC::DEVICE) {
        cudaMalloc(&ptr, sizeof(T) * num);
        cudaMemset(ptr, 0, sizeof(T) * num);
    }
}

template<class T>
void FieldFrame<T>::release() {
    if (loc == LOC::HOST) {
        free(ptr);
    } else if (loc == LOC::DEVICE) {
        cudaFree(ptr);
    }
    loc = LOC::NONE;
}

template<class T>
struct Field {
    FieldFrame<T>    host;
    FieldFrame<T>     dev;
    FieldFrame<T> *devptr;
    unsigned int     size;
    unsigned int      dim;
    unsigned int      num;
    unsigned int      loc;
    unsigned int    label;
    Field();
    Field(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel);
    Field(unsigned int vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel);
    ~Field();
    void init(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel);
    void init(unsigned int vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel);
    void release(unsigned int vloc);
    void sync(unsigned int direction);
    __host__ __device__ T& operator()(int idx) {return host(idx);}
    __host__ __device__ T& operator()(int idx, int dim_idx) {return host(idx, dim_idx);}
};

template<class T>
Field<T>::Field() : size(0), dim(0), num(0), loc(LOC::NONE), label(0) {}

template<class T>
Field<T>::Field(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel) : size(vsize.x * vsize.y * vsize.z), dim(vdim), num(vsize.x * vsize.y * vsize.z * vdim), loc(vloc), label(vlabel), host(vsize, vdim, vloc & LOC::HOST, vlabel), dev(vsize, vdim, vloc & LOC::DEVICE, vlabel), devptr(nullptr) {
    if (loc & LOC::DEVICE) {
        cudaMalloc(&devptr, sizeof(FieldFrame<T>));
        cudaMemcpy(devptr, &dev, sizeof(FieldFrame<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
Field<T>::Field(unsigned int vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel) : size(vsize), dim(vdim), num(vsize * vdim), loc(vloc), label(vlabel), host(vsize, vdim, vloc & LOC::HOST, vlabel), dev(vsize, vdim, vloc & LOC::DEVICE, vlabel), devptr(nullptr) {
    if (loc & LOC::DEVICE) {
        cudaMalloc(&devptr, sizeof(FieldFrame<T>));
        cudaMemcpy(devptr, &dev, sizeof(FieldFrame<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
Field<T>::~Field() {
    if (loc & LOC::DEVICE) {
        cudaFree(devptr);
    }
    loc = LOC::NONE;
}

template<class T>
void Field<T>::init(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel) {
    assert(loc == LOC::NONE);
    size  = vsize.x * vsize.y * vsize.z;
    dim   = vdim;
    num   = size * dim;
    loc   = vloc;
    label = vlabel;
    host.init(size, dim, loc & LOC::HOST, label);
    dev.init(size, dim, loc & LOC::DEVICE, label);
    if (loc & LOC::DEVICE) {
        cudaMalloc(&devptr, sizeof(FieldFrame<T>));
        cudaMemcpy(devptr, &dev, sizeof(FieldFrame<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
void Field<T>::init(unsigned int vsize, unsigned int vdim, unsigned int vloc, unsigned int vlabel) {
    assert(loc == LOC::NONE);
    size  = vsize;
    dim   = vdim;
    num   = size * dim;
    loc   = vloc;
    label = vlabel;
    host.init(size, dim, loc & LOC::HOST, label);
    dev.init(size, dim, loc & LOC::DEVICE, label);
    if (loc & LOC::DEVICE) {
        cudaMalloc(&devptr, sizeof(FieldFrame<T>));
        cudaMemcpy(devptr, &dev, sizeof(FieldFrame<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
void Field<T>::release(unsigned int vloc) {
    if (vloc & LOC::HOST) {
        assert(loc & LOC::HOST);
        host.release();
        loc &= (~LOC::HOST);
    }
    if (vloc & LOC::DEVICE) {
        assert(loc & LOC::DEVICE);
        dev.release();
        cudaFree(devptr);
        loc &= (~LOC::DEVICE);
    }
}

template<class T>
void Field<T>::sync(unsigned int direction) {
    if (direction == SYNC::H2D) {
        assert(loc & LOC::HOST);
        if (loc ==LOC::BOTH) {
            cudaMemcpy(dev.ptr, host.ptr, sizeof(T) * num, cudaMemcpyHostToDevice);
        } else if (loc == LOC::HOST) {
            cudaMalloc(&(dev.ptr), sizeof(T) * num);
            cudaMemcpy(dev.ptr, host.ptr, sizeof(T) * num, cudaMemcpyHostToDevice);
            dev.loc |= LOC::DEVICE;
            loc     |= LOC::DEVICE;
            cudaMalloc(&devptr, sizeof(FieldFrame<T>));
            cudaMemcpy(devptr, &dev, sizeof(FieldFrame<T>), cudaMemcpyHostToDevice);
        }
    } else if (direction == SYNC::D2H) {
        assert(loc & LOC::DEVICE);
        if (loc == LOC::BOTH) {
            cudaMemcpy(host.ptr, dev.ptr, sizeof(T) * num, cudaMemcpyDeviceToHost);
        } else if (loc == LOC::DEVICE) {
            host.ptr = (T*)malloc(sizeof(T) * num);
            cudaMemcpy(host.ptr, dev.ptr, sizeof(T) * num, cudaMemcpyDeviceToHost);
            host.loc |= LOC::HOST;
            loc      |= LOC::HOST;
        }
    }
}

// __global__ static void fscala_norm2sq_kernel(FieldFrame<double> &a, double *partial_sum, Mapper domain, Mapper range) {
//     unsigned int n_threads = block_size.x * block_size.y * block_size.z;
//     __shared__ double cache[n_threads];
//     unsigned int i, j, k;
//     UTIL::THREAD2IJK(i, j, k);
//     double value = 0;
//     unsigned int thread_idx = UTIL::IDX(threadIdx.x, threadIdx.y, threadIdx.z, blockDim);
//     if (i < range.size.x && j < range.size.y && k < range.size.z) {
//         i += range.offset.x;
//         j += range.offset.y;
//         k += range.offset.z;
//         unsigned int idx = UTIL::IDX(i, j, k, domain.size);
//         value = a(idx);
//     }
//     cache[thread_idx] = value * value;
//     __syncthreads();

//     unsigned int length = n_threads;
//     while (n_threads > 1) {
//         unsigned int cut = length / 2;
//         unsigned int reduce = length - cut;
//         if (thread_idx < cut) {
//             cache[thread_idx] += cache[thread_idx + reduce];
//         }
//         __syncthreads();
//         length = reduce;
//     }

//     if (thread_idx == 0) {
//         unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
//         partial_sum[block_idx] = cache[thread_idx];
//     }
// }

// static double fscala_norm2sq(Field<double> &a, Mapper &domain) {
//     assert(a.dim == 1);
//     dim3 &size = domain.size;
//     const unsigned int g = guide;
//     Mapper map(
//         dim3(size.x - 2 * g, size.y - 2 * g, size.z - 2 * g),
//         dim3(g, g, g)
//     );
//     double *partial_sum, *partial_sum_dev;
//     dim3 grid_size(
//         (map.size.x + block_size.x - 1) / block_size.x,
//         (map.size.y + block_size.y - 1) / block_size.y,
//         (map.size.z + block_size.z - 1) / block_size.z
//     );
//     unsigned int n_blocks = grid_size.x * grid_size.y * grid_size.z;
    
// }

namespace UTIL {

template<class T>
static void fieldcpy(Field<T> &dst, Field<T> &src, unsigned int loc) {
    assert(dst.num == src.num);
    if (loc & LOC::HOST) {
        assert(dst.loc & src.loc & LOC::HOST);
        memcpy(dst.host.ptr, src.host.ptr, sizeof(T) * dst.num);
    }
    if (loc & LOC::DEVICE) {
        assert(dst.loc & src.loc & LOC::DEVICE);
        cudaMemcpy(dst.dev.ptr, src.dev.ptr, sizeof(T) * dst.num, cudaMemcpyDeviceToDevice);
    }
}

template<class T>
static void fieldclear(Field<T> &dst, unsigned int loc) {
    if (loc & LOC::HOST) {
        assert(dst.loc & LOC::HOST);
        memset(dst.host.ptr, 0, sizeof(T) * dst.num);
    }
    if (loc & LOC::DEVICE) {
        assert(dst.loc & LOC::DEVICE);
        cudaMemset(dst.dev.ptr, 0, sizeof(T) * dst.num);
    }
}

}

}

#endif