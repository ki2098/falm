#ifndef FALM_MATRIX_H
#define FALM_MATRIX_H

#include <assert.h>
#include "typedef.h"
#include "flag.h"
#include "util.h"

namespace Falm {

template<typename T>
struct MatrixFrame {
    T               *ptr;
    uint2          shape;
    unsigned int    size;
    unsigned int hdctype;
    int            label;
    __host__ __device__ T &operator()(unsigned int _idx) {return ptr[_idx];}
    __host__ __device__ T &operator()(unsigned int _row, unsigned int _col) {return ptr[_row + _col * shape.x];}

    MatrixFrame() : ptr(nullptr), shape(unit2{0, 0}), size(0), hdctype(HDCTYPE::Empty), label(0) {}
    MatrixFrame(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label);
    MatrixFrame(unsigned int _row, unsigned int _col, unsigned int _hdctype, int _label);
    ~MatrixFrame();

    void init(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label);
    void init(unsigned int _row, unsigned int _col, unsigned int _hdctype, int _label);
    void release();
};

template<typename T> MatrixFrame<T>::MatrixFrame(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label) :
    shape(uint2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    label(_label)
{
    if (hdctype == HDCTYPE::Host) {
        ptr = falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCTYPE::Device) {
        ptr = falmDevMalloc(sizeof(T) * size);
        falmDevMemset(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> MatrixFrame<T>::MatrixFrame(unsigned int _row, unsigned int _col, unsigned int _hdctype, int _label) :
    shape(uint2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    label(_label)
{
    if (hdctype == HDCTYPE::Host) {
        ptr = falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCTYPE::Device) {
        ptr = falmDevMalloc(sizeof(T) * size);
        falmDevMemset(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> MatrixFrame<T>::~MatrixFrame() {
    if (hdctype == HDCTYPE::Host) {
        falmHostFreePtr(ptr);
    } else if (hdctype == HDCTYPE::Device) {
        falmDevFreePtr(ptr);
    }
    hdctype = HDCTYPE::Empty;
}

template<typename T> void MatrixFrame<T>::init(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCTYPE::Empty);
    shape   = uint2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype == HDCTYPE::Host) {
        ptr = falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCTYPE::Device) {
        ptr = falmDevMalloc(sizeof(T) * size);
        falmDevMemset(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> void MatrixFrame<T>::init(unsigned int _row, unsigned int _col, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCTYPE::Empty);
    shape   = uint2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype == HDCTYPE::Host) {
        ptr = falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCTYPE::Device) {
        ptr = falmDevMalloc(sizeof(T) * size);
        falmDevMemset(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> void MatrixFrame<T>::release() {
    if (hdctype == HDCTYPE::Host) {
        falmHostFreePtr(ptr);
    } else if (hdctype == HDCTYPE::Device) {
        falmDevFreePtr(ptr);
    }
    hdctype = HDCTYPE::Empty;
}


template<typename T>
struct Matrix {
    MatrixFrame<T>    host;
    MatrixFrame<T>  device;
    MatrixFrame<T> *devptr;
    uint2            shape;
    unsigned int      size;
    unsigned int   hdctype;
    int              label;

    __host__ __device__ T &operator()(unsigned int _idx) {return host(_idx);}
    __host__ __device__ T &operator()(unsigned int _row, unsigned int _col) {return host(_row, _col);}

    Matrix() : shape(uint2{0, 0}), size(0), hdctype(HDCTYPE::Empty), label(0), devptr(nullptr) {}
    Matrix(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label);
    Matrix(unsigned _row, unsigned int _col, unsigned int _hdctype, int _label);
    ~Matrix();

    void init(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label);
    void init(unsigned _row, unsigned int _col, unsigned int _hdctype, int _label);
    void release(unsigned int _hdctype);
    void sync(unsigned int _mcptype);
};

template<typename T> Matrix<T>::Matrix(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label) :
    host(_dom, _dim, _hdctype & HDCTYPE::Host, _label),
    device(_dom, _dim, _hdctype & HDCTYPE::Device, _label),
    shape(uint2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    label(_label)
{
    if (hdctype & HDCTYPE::Device) {
        devptr = falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &device, sizeof(MatrixFrame<T>), MCPTYPE::Hst2Dev);
    }
}

template<typename T> Matrix<T>::Matrix(unsigned _row, unsigned int _col, unsigned int _hdctype, int _label) :
    host(_row, _col, _hdctype & HDCTYPE::Host, _label),
    device(_row, _col, _hdctype & HDCTYPE::Device, _label),
    shape(uint2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    label(_label)
{
    if (hdctype & HDCTYPE::Device) {
        devptr = falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &device, sizeof(MatrixFrame<T>), MCPTYPE::Hst2Dev);
    }
}

template<typename T> Matrix<T>::~Matrix() {
    if (hdctype & HDCTYPE::Device) {
        falmDevFreePtr(devptr);
    }
    hdctype = HDCTYPE::Empty;
}

template<typename T> void Matrix<T>::init(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCTYPE::Empty);
    host.init(_dom, _dim, _hdctype & HDCTYPE::Host, _label);
    device.init(_dom, _dim, _hdctype & HDCTYPE::Device, _label);
    shape   = uint2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype & HDCTYPE::Device) {
        devptr = falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &device, sizeof(MatrixFrame<T>), MCPTYPE::Hst2Dev);
    }
}

template<typename T> void Matrix<T>::init(unsigned _row, unsigned int _col, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCTYPE::Empty);
    host.init(_row, _col, _hdctype & HDCTYPE::Host, _label);
    device.init(_row, _col, _hdctype & HDCTYPE::Device, _label);
    shape   = uint2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype & HDCTYPE::Device) {
        devptr = falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &device, sizeof(MatrixFrame<T>), MCPTYPE::Hst2Dev);
    }
}

template<typename T> void Matrix<T>::release(unsigned int _hdctype) {
    if (_hdctype & HDCTYPE::Host) {
        assert(hdctype & HDCTYPE::Host);
        host.release();
        hdctype &= ~(HDCTYPE::Host);
    }
    if (_hdctype & HDCTYPE::Device) {
        assert(hdctype & HDCTYPE::Device);
        device.release();
        hdctype &= ~(HDCTYPE::Device);
    }
}

template<typename T> void Matrix<T>::sync(unsigned int _mcptype) {
    if (_mcptype == MCPTYPE::Hst2Dev) {
        assert(hdctype & HDCTYPE::Host);
        if (hdctype & HDCTYPE::Device) {
            falmMemcpy(host.ptr, device.ptr, sizeof(T) * size, MCPTYPE::Hst2Dev);
        }
    }
}

}

#endif