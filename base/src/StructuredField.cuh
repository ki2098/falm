#ifndef _STRUCTURED_FIELD_H_
#define _STRUCTURED_FIELD_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include "Dom.cuh"
#include "param.h"

namespace FALMLoc {
static const int NONE   = 0;
static const int HOST   = 1;
static const int DEVICE = 2;
static const int BOTH   = HOST | DEVICE;
}

namespace FALM {

template<class T>
struct FieldCp {
    T  *_arr;
    int _row;
    int _col;
    int _num;
    int _loc;
    int _lab;
    FieldCp(dim3 &size, int col, int loc, int lab);
    FieldCp(int row, int col, int loc, int lab);
    FieldCp();
    ~FieldCp();
    void init(dim3 &size, int col, int loc, int lab);
    void init(int row, int col, int loc, int lab);
    void release();
    __host__ __device__ T& operator()(int idx) {return _arr[idx];}
    __host__ __device__ T& operator()(int row_idx, int col_idx) {return _arr[col_idx * _row + row_idx];}
};

template<class T>
FieldCp<T>::FieldCp() : _row(0), _col(0), _num(0), _loc(FALMLoc::NONE), _lab(0), _arr(nullptr) {/* printf("Default constructor of FieldCp called\n"); */}

template<class T>
FieldCp<T>::FieldCp(dim3 &size, int col, int loc, int lab) : _row(size.x * size.y * size.z), _col(col), _num(size.x * size.y * size.z * col), _loc(loc), _lab(lab) {
    if (loc == FALMLoc::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
    } else if (loc == FALMLoc::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
    }
}

template<class T>
FieldCp<T>::FieldCp(int row, int col, int loc, int lab) : _row(row), _col(col), _num(row * col), _loc(loc), _lab(lab) {
    if (loc == FALMLoc::HOST) {
        _arr = (T*)malloc(sizeof(T) * _num);
        memset(_arr, 0, sizeof(T) * _num);
    } else if (loc == FALMLoc::DEVICE) {
        cudaMalloc(&_arr, sizeof(T) * _num);
        cudaMemset(_arr, 0, sizeof(T) * _num);
    }
}

template<class T>
void FieldCp<T>::init(dim3 &size, int col, int loc, int lab) {
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
void FieldCp<T>::init(int row, int col, int loc, int lab) {
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

__global__ static void calc_norm2_kernel(FALM::FieldCp<double> &a, double *partial_sum, FALM::DomCp &dom) {
    __shared__ double cache[n_threads];
    unsigned int stride = get_global_size();
    double temp_sum = 0;
    for (unsigned int idx = get_global_idx(); idx < dom._inum; idx += stride) {

    }
}

}

#endif