#ifndef _STRUCTURED_FIELD_H_
#define _STRUCTURED_FIELD_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <mpi.h>
#include "Dom.cuh"
#include "param.h"

namespace FALM {

template<class T>
struct FieldCp {
    T  *_arr;
    unsigned int _row;
    unsigned int _col;
    unsigned int _num;
    unsigned int _loc;
    unsigned int _lab;
    FieldCp(dim3 &size, unsigned int col, unsigned int loc, unsigned int lab);
    FieldCp(unsigned int row, unsigned int col, unsigned int loc, unsigned int lab);
    FieldCp();
    ~FieldCp();
    void init(dim3 &size, unsigned int col, unsigned int loc, unsigned int lab);
    void init(unsigned int row, unsigned int col, unsigned int loc, unsigned int lab);
    void release();
    __host__ __device__ T& operator()(unsigned int idx) {return _arr[idx];}
    __host__ __device__ T& operator()(unsigned int row_idx, unsigned int col_idx) {return _arr[col_idx * _row + row_idx];}
};

template<class T>
FieldCp<T>::FieldCp() : _row(0), _col(0), _num(0), _loc(FALMLoc::NONE), _lab(0), _arr(nullptr) {/* printf("Default constructor of FieldCp called\n"); */}

template<class T>
FieldCp<T>::FieldCp(dim3 &size, unsigned int col, unsigned int loc, unsigned int lab) : _row(size.x * size.y * size.z), _col(col), _num(size.x * size.y * size.z * col), _loc(loc), _lab(lab) {
    if (loc == FALMLoc::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
    } else if (loc == FALMLoc::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
    }
}

template<class T>
FieldCp<T>::FieldCp(unsigned int row, unsigned int col, unsigned int loc, unsigned int lab) : _row(row), _col(col), _num(row * col), _loc(loc), _lab(lab) {
    if (loc == FALMLoc::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
    } else if (loc == FALMLoc::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
    }
}

template<class T>
void FieldCp<T>::init(dim3 &size, unsigned int col, unsigned int loc, unsigned int lab) {
    assert(_loc == FALMLoc::NONE);
    _row = size.x * size.y * size.z;
    _col = col;
    _num = _row * _col;
    _loc = loc;
    _lab = lab;
    if (loc == FALMLoc::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
        // printf("initializer of FieldCp %d called to free on HOST\n", _lab);
    } else if (loc == FALMLoc::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
        // printf("initializer of FieldCp %d called to free on DEVICE\n", _lab);
    }
}

template<class T>
void FieldCp<T>::init(unsigned int row, unsigned int col, unsigned int loc, unsigned int lab) {
    assert(_loc == FALMLoc::NONE);
    _row = row;
    _col = col;
    _num = _row * _col;
    _loc = loc;
    _lab = lab;
    if (loc == FALMLoc::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
        // printf("initializer of FieldCp %d called to init on HOST\n", _lab);
    } else if (loc == FALMLoc::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
        // printf("initializer of FieldCp %d called to init on DEVICE\n", _lab);
    }
}

template<class T>
void FieldCp<T>::release() {
    if (_loc == FALMLoc::HOST) {
        // printf("release of FieldCp %d called to free on HOST\n", _lab);
        free(_arr);
    } else if (_loc == FALMLoc::DEVICE) {
        // printf("release of FieldCp %d called to free on DEVICE\n", _lab);
        cudaFree(_arr);
    }
    _loc = FALMLoc::NONE;
}

template<class T>
FieldCp<T>::~FieldCp() {
    if (_loc == FALMLoc::HOST) {
        // printf("destructor of FieldCp %d called to free on HOST\n", _lab);
        free(_arr);
        _loc &= (~FALMLoc::HOST);
    } else if (_loc == FALMLoc::DEVICE) {
        // printf("destructor of FieldCp %d called to free on DEVICE\n", _lab);
        cudaFree(_arr);
        _loc &= (~FALMLoc::DEVICE);
    }
}

template<class T>
struct Field {
    FieldCp<T>  _hh;
    FieldCp<T>  _hd;
    FieldCp<T> *_dd;
    int         _row;
    int         _col;
    int         _num;
    int         _loc;
    int         _lab;
    Field(dim3 &size, int col, int loc, int lab);
    Field(int row, int col, int loc, int lab);
    Field();
    ~Field();
    void init(dim3 &size, int col, int loc, int lab);
    void init(int row, int col, int loc, int lab);
    void release(int loc);
    void sync_h2d();
    void sync_d2h();
    __host__ __device__ T& operator()(int idx) {return _hh(idx);}
    __host__ __device__ T& operator()(int row_idx, int col_idx) {return _hh(row_idx, col_idx);}
};

template<class T>
Field<T>::Field() : _row(0), _col(0), _num(0), _loc(FALMLoc::NONE), _lab(0), _dd(nullptr) {/* printf("Default constructor of Field called\n"); */}

template<class T>
Field<T>::Field(dim3 &size, int col, int loc, int lab) : _row(size.x * size.y * size.z), _col(col), _num(size.x * size.y * size.z * col), _loc(loc), _lab(lab), _hh(size, col, (loc & FALMLoc::HOST), lab), _hd(size, col, (loc & FALMLoc::DEVICE), lab), _dd(nullptr) {
    if (loc & FALMLoc::DEVICE) {
        cudaMalloc(&_dd, sizeof(FieldCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(FieldCp<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
Field<T>::Field(int row, int col, int loc, int lab) : _row(row), _col(col), _num(row * col), _loc(loc), _lab(lab), _hh(row, col, (loc & FALMLoc::HOST), lab), _hd(row, col, (loc & FALMLoc::DEVICE), lab), _dd(nullptr) {
    if (loc & FALMLoc::DEVICE) {
        cudaMalloc(&_dd, sizeof(FieldCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(FieldCp<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
void Field<T>::init(dim3 &size, int col, int loc, int lab) {
    assert(_loc == FALMLoc::NONE);
    _row = size.x * size.y * size.z;
    _col = col;
    _num = _row * _col;
    _loc = loc;
    _lab = lab;
    _hh.init(_row, _col, _loc & FALMLoc::HOST  , _lab);
    _hd.init(_row, _col, _loc & FALMLoc::DEVICE, _lab);
    if (loc & FALMLoc::DEVICE) {
        cudaMalloc(&_dd, sizeof(FieldCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(FieldCp<T>), cudaMemcpyHostToDevice);
        // printf("initializer of Field %d called to init on DEVICE\n", _lab);
    }
}

template<class T>
void Field<T>::init(int row, int col, int loc, int lab) {
    assert(_loc == FALMLoc::NONE);
    _row = row;
    _col = col;
    _num = _row * _col;
    _loc = loc;
    _lab = lab;
    _hh.init(_row, _col, _loc & FALMLoc::HOST  , _lab);
    _hd.init(_row, _col, _loc & FALMLoc::DEVICE, _lab);
    if (loc & FALMLoc::DEVICE) {
        cudaMalloc(&_dd, sizeof(FieldCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(FieldCp<T>), cudaMemcpyHostToDevice);
        // printf("initializer of Field %d called to init on DEVICE\n", _lab);
    }
}

template<class T>
void Field<T>::release(int loc) {
    if ((loc & FALMLoc::HOST) && (_loc & FALMLoc::HOST)) {
        // printf("release of Field %d called to free on HOST\n", _lab);
        _hh.release();
        _loc &= (~FALMLoc::HOST);
    }
    if ((loc & FALMLoc::DEVICE) && (_loc & FALMLoc::DEVICE)) {
        // printf("release of Field %d called to free on DEVICE\n", _lab);
        _hd.release();
        cudaFree(_dd);
        _loc &= (~FALMLoc::DEVICE);
    }

}

template<class T>
Field<T>::~Field() {
    if (_loc & FALMLoc::DEVICE) {
        // printf("destructor of Field %d called to free on DEVICE\n", _lab);
        cudaFree(_dd);
    }
    _loc = FALMLoc::NONE;
}

template<class T>
void Field<T>::sync_h2d() {
    if (_loc == FALMLoc::BOTH) {
        cudaMemcpy(_hd._arr, _hh._arr, sizeof(T) * _hh._num, cudaMemcpyHostToDevice);
    } else if (_loc == FALMLoc::HOST) {
        cudaMalloc(&(_hd._arr), sizeof(T) * _num);
        cudaMemcpy(_hd._arr, _hh._arr, sizeof(T) * _num, cudaMemcpyHostToDevice);
        _hd._loc |= FALMLoc::DEVICE;
        _loc     |= FALMLoc::DEVICE;
        cudaMalloc(&_dd, sizeof(FieldCp<T>));
        cudaMemcpy(_dd, &_hd, sizeof(FieldCp<T>), cudaMemcpyHostToDevice);
    }
}

template<class T>
void Field<T>::sync_d2h() {
    if (_loc == FALMLoc::BOTH) {
        cudaMemcpy(_hh._arr, _hd._arr, sizeof(T) * _num, cudaMemcpyDeviceToHost);
    } else if (_loc == FALMLoc::DEVICE) {
        _hh._arr = (T*)malloc(sizeof(T) * _num);
        cudaMemcpy(_hh._arr, _hd._arr, sizeof(T) * _num, cudaMemcpyDeviceToHost);
        _hh._loc |= FALMLoc::HOST;
        _loc     |= FALMLoc::HOST;
    }
}

}

namespace FALMUtil {

__global__ static void fscala_norm2_kernel(FALM::FieldCp<double> &a, double *partial_sum, FALM::Dom dom, FALM::Dom mapper) {
    __shared__ double cache[n_threads];
    unsigned int stride = get_global_size();
    double temp_sum = 0;
    for (unsigned int idx = get_global_idx(); idx < mapper._num; idx += stride) {
        unsigned int ii, ij, ik;
        FALMUtil::d123(idx, ii, ij, ik, mapper._size);
        unsigned int oi, oj, ok;
        oi = ii + mapper._offset.x;
        oj = ij + mapper._offset.y;
        ok = ik + mapper._offset.z;
        unsigned int odx = FALMUtil::d321(oi, oj, ok, dom._size);
        double value = a(odx);
        temp_sum += value * value;
    }
    cache[threadIdx.x] = temp_sum;
    __syncthreads();

    int length = n_threads;
    while (length > 1) {
        int cut = length / 2;
        int reduce = length - cut;
        if (threadIdx.x < cut) {
            cache[threadIdx.x] += cache[threadIdx.x + reduce];
        }
        __syncthreads();
        length = reduce;
    }

    if (threadIdx.x == 0) {
        partial_sum[blockIdx.x] = cache[0];
    }
}

static double fscala_norm2(FALM::Field<double> &a, FALM::Dom &dom) {
    assert(a._col == 1);
    dim3 &sz             = dom._size;
    const unsigned int g = FALM::guide;
    FALM::Dom inner(sz.x - 2 * g, sz.y - 2 * g, sz.z - 2 * g, g, g, g);
    double *partial_sum, *partial_sum_dev;
    cudaMalloc(&partial_sum_dev, sizeof(double) * n_blocks);
    partial_sum = (double*)malloc(sizeof(double) * n_blocks);

    fscala_norm2_kernel<<<n_blocks, n_threads>>>(*(a._dd), partial_sum_dev, dom, inner);

    cudaMemcpy(partial_sum, partial_sum_dev, sizeof(double) * n_blocks, cudaMemcpyDeviceToHost);

    double sum = partial_sum[0];
    for (int i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    free(partial_sum);
    cudaFree(partial_sum_dev);

    return sqrt(sum);
}

__global__ static void fscala_sum_kernel(FALM::FieldCp<double> &a, double *partial_sum, FALM::Dom dom, FALM::Dom mapper) {
    __shared__ double cache[n_threads];
    unsigned int stride = get_global_size();
    double temp_sum = 0;
    for (unsigned int idx = get_global_idx(); idx < mapper._num; idx += stride) {
        unsigned int ii, ij, ik;
        FALMUtil::d123(idx, ii, ij, ik, mapper._size);
        unsigned int oi, oj, ok;
        oi = ii + mapper._size.x;
        oj = ij + mapper._size.y;
        ok = ik + mapper._size.z;
        unsigned int odx = FALMUtil::d321(oi, oj, ok, dom._size);
        double value = a(odx);
        temp_sum += value;
    }
    cache[threadIdx.x] = temp_sum;
    __syncthreads();

    int length = n_threads;
    while (length > 1) {
        int cut = length / 2;
        int reduce = length - cut;
        if (threadIdx.x < cut) {
            cache[threadIdx.x] += cache[threadIdx.x + reduce];
        }
        __syncthreads();
        length = reduce;
    }

    if (threadIdx.x == 0) {
        partial_sum[blockIdx.x] = cache[0];
    }
}

static double fscala_sum(FALM::Field<double> &a, FALM::Dom &dom) {
    assert(a._col == 1);
    dim3             &sz = dom._size;
    const unsigned int g = FALM::guide;
    FALM::Dom inner(sz.x - 2 * g, sz.y - 2 * g, sz.z - 2 * g, g, g, g);
    double *partial_sum, *partial_sum_dev;
    cudaMalloc(&partial_sum_dev, sizeof(double) * n_blocks);
    partial_sum = (double*)malloc(sizeof(double) * n_blocks);

    fscala_sum_kernel<<<n_blocks, n_threads>>>(*(a._dd), partial_sum_dev, dom, inner);

    cudaMemcpy(partial_sum, partial_sum_dev, sizeof(double) * n_blocks, cudaMemcpyDeviceToHost);

    double sum = partial_sum[0];
    for (int i = 1; i < n_blocks; i ++) {
        sum += partial_sum[i];
    }

    free(partial_sum);
    cudaFree(partial_sum_dev);

    return sum;
}

__global__ static void fscala_zero_avg_kernel(FALM::FieldCp<double> &a, FALM::Dom dom, FALM::Dom mapper, double avg) {
    unsigned int stride = FALMUtil::get_global_size();
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < mapper._num; idx += stride) {
        unsigned int ii, ij, ik;
        FALMUtil::d123(idx, ii, ij, ik, mapper._size);
        unsigned int oi, oj, ok;
        oi = ii + mapper._size.x;
        oj = ij + mapper._size.y;
        ok = ik + mapper._size.z;
        unsigned int odx = FALMUtil::d321(oi, oj, ok, dom._size);
        a(odx) -= avg;
    }
}

static void fscala_zero_avg(FALM::Field<double> &a, FALM::Dom &dom, FALM::Dom &global, int mpi_size, int mpi_rank) {
    double sum = fscala_sum(a, dom);
    if (mpi_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    dim3 &gsz = global._size;
    const unsigned int g = FALM::guide;
    unsigned int gnum = (gsz.x - 2 * g) * (gsz.y - 2 * g) * (gsz.z - 2 * g);
    double avg = sum / gnum;
    dim3 &sz = dom._size;
    FALM::Dom inner(sz.x - 2 * g, sz.y - 2 * g, sz.z - 2 * g, g, g, g);
    fscala_zero_avg_kernel<<<n_blocks, n_threads>>>(*(a._dd), dom, inner, avg);
}

template<class T>
static void field_cpy(FALM::Field<T> &dst, FALM::Field<T> &src, unsigned int loc) {
    assert(dst._num == src._num);
    if (loc & FALMLoc::HOST) {
        assert(dst._loc & src._loc & FALMLoc::HOST);
        memcpy(dst._hh._arr, src._hh._arr, sizeof(T) * dst._num);
    }
    if (loc & FALMLoc::DEVICE) {
        assert(dst._loc & src._loc & FALMLoc::DEVICE);
        cudaMemcpy(dst._hd._arr, src._hd._arr, sizeof(T) * dst._num, cudaMemcpyDeviceToDevice);
    }
}

template<class T>
static void field_clear(FALM::Field<T> &dst, unsigned int loc) {
    if (loc & FALMLoc::HOST) {
        assert(dst._loc & FALMLoc::HOST);
        memset(dst._hh._arr, 0, sizeof(T) * dst._num);
    }
    if (loc & FALMLoc::DEVICE) {
        assert(dst._loc & FALMLoc::DEVICE);
        cudaMemset(dst._hd._arr, 0, sizeof(T) * dst._num);
    }
}

}

#endif