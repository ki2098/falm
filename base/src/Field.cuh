#ifndef FALM_FIELD_CUH
#define FALM_FIELD_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <mpi.h>
#include "Util.cuh"

namespace FALM {

template<class T>
struct FieldFrame {
    T           *arr;
    unsigned int num;
    unsigned int dim;
    unsigned int len;
    unsigned int loc;
    unsigned int lab;
    FieldFrame();
    FieldFrame(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlab);
    FieldFrame(unsigned int vnum, unsigned int vdim, unsigned int vloc, unsigned int vlab);
    ~FieldFrame();
    void init(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlab);
    void init(unsigned int vnum, unsigned int vdim, unsigned int vloc, unsigned int vlab);
    void release();
    __host__ __device__ T& operator()(unsigned int idx) {return arr[idx];}
    __host__ __device__ T& operator()(unsigned int idx, unsigned int d) {return arr[idx + num * d];}
};

template<class T>
FieldFrame<T>::FieldFrame() : num(0), dim(0), len(0), loc(LOC::NONE), lab(0), arr(nullptr) {}

template<class T>
FieldFrame<T>::FieldFrame(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlab) : num(vsize.x * vsize.y * vsize.z), dim(vdim), len(vsize.x * vsize.y * vsize.z * vdim), loc(vloc), lab(vlab) {
    if (loc == LOC::HOST) {
        arr = (T*)malloc(sizeof(T) * len);
        memset(arr, 0, sizeof(T) * len);
    } else if (loc == LOC::DEVICE) {
        cudaMalloc(&arr, sizeof(T) * len);
        cudaMemset(arr, 0, sizeof(T) * len);
    }
}

template<class T>
FieldFrame<T>::FieldFrame(unsigned int vnum, unsigned int vdim, unsigned int vloc, unsigned int vlab) : num(vnum), dim(vdim), len(vnum * vdim), loc(vloc), lab(vlab) {
    if (loc == LOC::HOST) {
        arr = (T*)malloc(sizeof(T) * len);
        memset(arr, 0, sizeof(T) * len);
    } else if (loc == LOC::DEVICE) {
        cudaMalloc(&arr, sizeof(T) * len);
        cudaMemset(arr, 0, sizeof(T) * len);
    }
}

template<class T>
FieldFrame<T>::~FieldFrame() {
    if (loc == LOC::HOST) {
        free(arr);
        loc &= (~LOC::HOST);
    } else if (loc == LOC::DEVICE) {
        cudaFree(arr);
        loc &= (~LOC::DEVICE);
    }
}

template<class T>
void FieldFrame<T>::init(dim3 &vsize, unsigned int vdim, unsigned int vloc, unsigned int vlab) {
    num = vsize.x * vsize.y * vsize.z;
    dim = vdim;
    len = num * dim;
    loc = vloc;
    lab = vlab;
    if (loc == LOC::HOST) {
        arr = (T*)malloc(sizeof(T) * len);
        memset(arr, 0, sizeof(T) * len);
    } else if (loc == LOC::DEVICE) {
        cudaMalloc(&arr, sizeof(T) * len);
        cudaMemset(arr, 0, sizeof(T) * len);
    }
}

}


#endif