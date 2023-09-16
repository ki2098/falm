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

    MatrixFrame() : ptr(nullptr), shape(uint2{0, 0}), size(0), hdctype(HDCType::Empty), label(0) {}
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
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
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
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
        falmDevMemset(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> MatrixFrame<T>::~MatrixFrame() {
    if (hdctype == HDCType::Host) {
        falmHostFreePtr(ptr);
    } else if (hdctype == HDCType::Device) {
        falmDevFreePtr(ptr);
    }
    ptr = nullptr;
    hdctype = HDCType::Empty;
}

template<typename T> void MatrixFrame<T>::init(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCType::Empty);
    shape   = uint2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
        falmDevMemset(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> void MatrixFrame<T>::init(unsigned int _row, unsigned int _col, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCType::Empty);
    shape   = uint2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
        falmHostMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
        falmDevMemset(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> void MatrixFrame<T>::release() {
    if (hdctype == HDCType::Host) {
        falmHostFreePtr(ptr);
    } else if (hdctype == HDCType::Device) {
        falmDevFreePtr(ptr);
    }
    ptr = nullptr;
    hdctype = HDCType::Empty;
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

    Matrix() : shape(uint2{0, 0}), size(0), hdctype(HDCType::Empty), label(0), devptr(nullptr) {}
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
    host(_dom, _dim, _hdctype & HDCType::Host, _label),
    dev(_dom, _dim, _hdctype & HDCType::Device, _label),
    shape(uint2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    label(_label)
{
    if (hdctype & HDCType::Device) {
        devptr = (MatrixFrame<T>*)falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
    }
}

template<typename T> Matrix<T>::Matrix(unsigned _row, unsigned int _col, unsigned int _hdctype, int _label) :
    host(_row, _col, _hdctype & HDCType::Host, _label),
    dev(_row, _col, _hdctype & HDCType::Device, _label),
    shape(uint2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    label(_label)
{
    if (hdctype & HDCType::Device) {
        devptr = (MatrixFrame<T>*)falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
    }
}

template<typename T> Matrix<T>::~Matrix() {
    if (hdctype & HDCType::Device) {
        falmDevFreePtr(devptr);
    }
    devptr = nullptr;
    hdctype = HDCType::Empty;
}

template<typename T> void Matrix<T>::init(uint3 _dom, unsigned int _dim, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCType::Empty);
    host.init(_dom, _dim, _hdctype & HDCType::Host, _label);
    dev.init(_dom, _dim, _hdctype & HDCType::Device, _label);
    shape   = uint2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype & HDCType::Device) {
        devptr = (MatrixFrame<T>*)falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
    }
}

template<typename T> void Matrix<T>::init(unsigned _row, unsigned int _col, unsigned int _hdctype, int _label) {
    assert(hdctype == HDCType::Empty);
    host.init(_row, _col, _hdctype & HDCType::Host, _label);
    dev.init(_row, _col, _hdctype & HDCType::Device, _label);
    shape   = uint2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    label   = _label;
    if (hdctype & HDCType::Device) {
        devptr = (MatrixFrame<T>*)falmDevMalloc(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
    }
}

template<typename T> void Matrix<T>::release(unsigned int _hdctype) {
    if (_hdctype & HDCType::Host) {
        assert(hdctype & HDCType::Host);
        host.release();
        hdctype &= ~(HDCType::Host);
    }
    if (_hdctype & HDCType::Device) {
        assert(hdctype & HDCType::Device);
        dev.release();
        falmDevFreePtr(devptr);
        devptr = nullptr;
        hdctype &= ~(HDCType::Device);
    }
}

template<typename T> void Matrix<T>::sync(unsigned int _mcptype) {
    if (_mcptype == MCpType::Hst2Dev) {
        assert(hdctype & HDCType::Host);
        if (hdctype & HDCType::Device) {
            falmMemcpy(dev.ptr, host.ptr, sizeof(T) * size, MCpType::Hst2Dev);
        } else {
            dev.init(shape.x, shape.y, HDCType::Device, label);
            falmMemcpy(dev.ptr, host.ptr, sizeof(T) * size, MCpType::Hst2Dev);
            devptr = (MatrixFrame<T>*)falmDevMalloc(sizeof(MatrixFrame<T>));
            falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
            hdctype |= HDCType::Device;
        }
    } else if (_mcptype == MCpType::Dev2Hst) {
        assert(hdctype & HDCType::Device);
        if (hdctype & HDCType::Host) {
            falmMemcpy(host.ptr, dev.ptr, sizeof(T) * size, MCpType::Dev2Hst);
        } else {
            host.init(shape.x, shape.y, HDCType::Host, label);
            falmMemcpy(host.ptr, dev.ptr, sizeof(T) * size, MCpType::Dev2Hst);
            hdctype |= HDCType::Host;
        }
    }
}

template<typename T> void Matrix<T>::cpy(Matrix<T> &src, unsigned int _hdctype) {
    if (_hdctype & HDCType::Host) {
        assert((hdctype & src.hdctype & HDCType::Host) && (size == src.size));
        falmMemcpy(host.ptr, src.host.ptr, sizeof(T) * size, MCpType::Hst2Hst);
    }
    if (_hdctype & HDCType::Device) {
        assert((hdctype & src.hdctype & HDCType::Device) && (size == src.size));
        falmMemcpy(dev.ptr, src.dev.ptr, sizeof(T) * size, MCpType::Dev2Dev);
    }
}

template<typename T> void Matrix<T>::clear(unsigned int _hdctype) {
    if (_hdctype & HDCType::Host) {
        assert(hdctype & HDCType::Host);
        falmHostMemset(host.ptr, 0, sizeof(T) * size);
    }
    if (_hdctype & HDCType::Device) {
        assert(hdctype & HDCType::Device);
        falmDevMemset(dev.ptr, 0, sizeof(T) * size);
    }
}

}

#endif