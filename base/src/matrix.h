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

    MatrixFrame() : ptr(nullptr), shape(uint2{0, 0}), size(0), hdctype(HDCTYPE::Empty), label(0) {}
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
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCTYPE::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
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
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCTYPE::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
        falmDevMemset(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> MatrixFrame<T>::~MatrixFrame() {
    if (hdctype == HDCTYPE::Host) {
        falmHostFreePtr(ptr);
    } else if (hdctype == HDCTYPE::Device) {
        falmDevFreePtr(ptr);
    }
    ptr = nullptr;
    hdctype = HDCTYPE::Empty;
}

template<typename T> void MatrixFrame<T>::init(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCTYPE::Empty);
    shape   = uint2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype == HDCTYPE::Host) {
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCTYPE::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
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
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCTYPE::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
        falmDevMemset(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> void MatrixFrame<T>::release() {
    if (hdctype == HDCTYPE::Host) {
        falmHostFreePtr(ptr);
    } else if (hdctype == HDCTYPE::Device) {
        falmDevFreePtr(ptr);
    }
    ptr = nullptr;
    hdctype = HDCTYPE::Empty;
}


template<typename T>
struct Matrix {
    MatrixFrame<T>    host;
    MatrixFrame<T>     dev;
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

    void cpy(Matrix<T> &src, unsigned int _hdctype);
    void clear(unsigned int _hdctype);
};

template<typename T> Matrix<T>::Matrix(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label) :
    host(_dom, _dim, _hdctype & HDCTYPE::Host, _label),
    dev(_dom, _dim, _hdctype & HDCTYPE::Device, _label),
    shape(uint2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    label(_label)
{
    if (hdctype & HDCTYPE::Device) {
        devptr = (MatrixFrame<T>*)falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCPTYPE::Hst2Dev);
    }
}

template<typename T> Matrix<T>::Matrix(unsigned _row, unsigned int _col, unsigned int _hdctype, int _label) :
    host(_row, _col, _hdctype & HDCTYPE::Host, _label),
    dev(_row, _col, _hdctype & HDCTYPE::Device, _label),
    shape(uint2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    label(_label)
{
    if (hdctype & HDCTYPE::Device) {
        devptr = (MatrixFrame<T>*)falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCPTYPE::Hst2Dev);
    }
}

template<typename T> Matrix<T>::~Matrix() {
    if (hdctype & HDCTYPE::Device) {
        falmDevFreePtr(devptr);
    }
    devptr = nullptr;
    hdctype = HDCTYPE::Empty;
}

template<typename T> void Matrix<T>::init(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCTYPE::Empty);
    host.init(_dom, _dim, _hdctype & HDCTYPE::Host, _label);
    dev.init(_dom, _dim, _hdctype & HDCTYPE::Device, _label);
    shape   = uint2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype & HDCTYPE::Device) {
        devptr = (MatrixFrame<T>*)falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCPTYPE::Hst2Dev);
    }
}

template<typename T> void Matrix<T>::init(unsigned _row, unsigned int _col, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCTYPE::Empty);
    host.init(_row, _col, _hdctype & HDCTYPE::Host, _label);
    dev.init(_row, _col, _hdctype & HDCTYPE::Device, _label);
    shape   = uint2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype & HDCTYPE::Device) {
        devptr = (MatrixFrame<T>*)falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCPTYPE::Hst2Dev);
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
        dev.release();
        falmDevFreePtr(devptr);
        devptr = nullptr;
        hdctype &= ~(HDCTYPE::Device);
    }
}

template<typename T> void Matrix<T>::sync(unsigned int _mcptype) {
    if (_mcptype == MCPTYPE::Hst2Dev) {
        assert(hdctype & HDCTYPE::Host);
        if (hdctype & HDCTYPE::Device) {
            falmMemcpy(dev.ptr, host.ptr, sizeof(T) * size, MCPTYPE::Hst2Dev);
        } else {
            dev.init(shape.x, shape.y, HDCTYPE::Device, label);
            falmMemcpy(dev.ptr, host.ptr, sizeof(T) * size, MCPTYPE::Hst2Dev);
            devptr = (MatrixFrame<T>*)falmDevMalloc(sizeof(MatrixFrame<T>));
            falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCPTYPE::Hst2Dev);
            hdctype |= HDCTYPE::Device;
        }
    } else if (_mcptype == MCPTYPE::Dev2Hst) {
        assert(hdctype & HDCTYPE::Device);
        if (hdctype & HDCTYPE::Host) {
            falmMemcpy(host.ptr, dev.ptr, sizeof(T) * size, MCPTYPE::Dev2Hst);
        } else {
            host.init(shape.x, shape.y, HDCTYPE::Host, label);
            falmMemcpy(host.ptr, dev.ptr, sizeof(T) * size, MCPTYPE::Dev2Hst);
            hdctype |= HDCTYPE::Host;
        }
    }
}

template<typename T> void Matrix<T>::cpy(Matrix<T> &src, unsigned int _hdctype) {
    if (_hdctype & HDCTYPE::Host) {
        assert((hdctype & src.hdctype & HDCTYPE::Host) && (size == src.size));
        falmMemcpy(host.ptr, src.host.ptr, sizeof(T) * size, MCPTYPE::Hst2Hst);
    }
    if (_hdctype & HDCTYPE::Device) {
        assert((hdctype & src.hdctype & HDCTYPE::Device) && (size == src.size));
        falmMemcpy(dev.ptr, src.dev.ptr, sizeof(T) * size, MCPTYPE::Dev2Dev);
    }
}

}

#endif