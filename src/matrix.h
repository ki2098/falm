#ifndef FALM_MATRIX_H
#define FALM_MATRIX_H

#include <assert.h>
#include <string>
#include "util.h"

namespace Falm {

enum class StencilMatrix {Empty, D3P7, D3P13, D2P5, D2P9, D1P3, D1P5};

template<typename T>
struct MatrixFrame {
    T                *ptr;
    INT2            shape;
    INT              size;
    FLAG          hdctype;
    StencilMatrix stencil;
    __host__ __device__ T &operator()(INT _idx) const {return ptr[_idx];}
    __host__ __device__ T &operator()(INT _row, INT _col) const {return ptr[_row + _col * shape[0]];}

    MatrixFrame(const MatrixFrame<T> &_mat) = delete;
    MatrixFrame<T>& operator=(const MatrixFrame<T> &_mat) = delete;

    MatrixFrame() : ptr(nullptr), shape(INT2{0, 0}), size(0), hdctype(HDCType::Empty), stencil(StencilMatrix::Empty) {}
    MatrixFrame(INT3 _dom, INT _dim, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    MatrixFrame(INT _row, INT _col, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    ~MatrixFrame();

    void alloc(INT3 _dom, INT _dim, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    void alloc(INT _row, INT _col, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    void release();
    void clear() {
        if (hdctype == HDCType::Host) {
            falmMemset(ptr, 0, sizeof(T) * size);
        } else if (hdctype == HDCType::Device) {
            falmMemsetDevice(ptr, 0, sizeof(T) * size);
        }
    }
};

template<typename T> MatrixFrame<T>::MatrixFrame(INT3 _dom, INT _dim, FLAG _hdctype, StencilMatrix _stencil) :
    shape(INT2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    stencil(_stencil)
{
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmMallocPinned(sizeof(T) * size);
        falmMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmMallocDevice(sizeof(T) * size);
        falmMemsetDevice(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> MatrixFrame<T>::MatrixFrame(INT _row, INT _col, FLAG _hdctype, StencilMatrix _stencil) :
    shape(INT2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    stencil(_stencil)
{
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmMallocPinned(sizeof(T) * size);
        falmMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmMallocDevice(sizeof(T) * size);
        falmMemsetDevice(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> MatrixFrame<T>::~MatrixFrame() {
    if (hdctype == HDCType::Host) {
        falmFreePinned(ptr);
    } else if (hdctype == HDCType::Device) {
        falmFreeDevice(ptr);
    }
    ptr = nullptr;
    hdctype = HDCType::Empty;
}

template<typename T> void MatrixFrame<T>::alloc(INT3 _dom, INT _dim, FLAG _hdctype, StencilMatrix _stencil) {
    assert(hdctype == HDCType::Empty);
    shape   = INT2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    stencil = _stencil;
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmMallocPinned(sizeof(T) * size);
        falmMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmMallocDevice(sizeof(T) * size);
        falmMemsetDevice(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> void MatrixFrame<T>::alloc(INT _row, INT _col, FLAG _hdctype, StencilMatrix _stencil) {
    assert(hdctype == HDCType::Empty);
    shape   = INT2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    stencil = _stencil;
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmMallocPinned(sizeof(T) * size);
        falmMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmMallocDevice(sizeof(T) * size);
        falmMemsetDevice(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> void MatrixFrame<T>::release() {
    if (hdctype == HDCType::Host) {
        falmFreePinned(ptr);
    } else if (hdctype == HDCType::Device) {
        falmFreeDevice(ptr);
    }
    ptr = nullptr;
    hdctype = HDCType::Empty;
}


template<typename T>
struct Matrix {

    MatrixFrame<T>    host;
    MatrixFrame<T>     dev;
    MatrixFrame<T> *devptr;

    INT2             shape;
    INT               size;
    FLAG           hdctype;
    std::string       name;
    StencilMatrix    stencil;

    __host__ __device__ T &operator()(INT _idx) const {return host(_idx);}
    __host__ __device__ T &operator()(INT _row, INT _col) const {return host(_row, _col);}

    Matrix(const Matrix<T> &_mat) = delete;
    Matrix<T>& operator=(const Matrix<T> &_mat) = delete;

    Matrix(std::string _name = "") : shape(INT2{0, 0}), size(0), hdctype(HDCType::Empty), name(_name), devptr(nullptr), stencil(StencilMatrix::Empty) {}
    Matrix(INT3 _dom, INT _dim, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil = StencilMatrix::Empty);
    Matrix(INT _row, INT _col, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil = StencilMatrix::Empty);
    Matrix(INT3 _dom, INT _dim, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty) : Matrix(_dom, _dim, _hdctype, "", _stencil) {}
    Matrix(INT _row, INT _col, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty) : Matrix(_row, _col, _hdctype, "", _stencil) {}
    
    ~Matrix();

    void alloc(INT3 _dom, INT _dim, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil = StencilMatrix::Empty);
    void alloc(INT _row, INT _col, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil = StencilMatrix::Empty);
    void alloc(INT3 _dom, INT _dim, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty) {
        alloc(_dom, _dim, _hdctype, name, _stencil);
    }
    void alloc(INT _row, INT _col, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty) {
        alloc(_row, _col, _hdctype, name, _stencil);
    }
    void release(FLAG _hdctype);
    void sync(FLAG _mcptype);

    void copy(Matrix<T> &src, FLAG _hdctype);
    void clear(FLAG _hdctype);

    const char *cname() {return name.c_str();}
};

template<typename T> Matrix<T>::Matrix(INT3 _dom, INT _dim, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil) :
    host(_dom, _dim, _hdctype & HDCType::Host, _stencil),
    dev(_dom, _dim, _hdctype & HDCType::Device, _stencil),
    shape(INT2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    name(_name),
    stencil(_stencil)
{
    if (hdctype & HDCType::Device) {
        devptr = (MatrixFrame<T>*)falmMallocDevice(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
    }
}

template<typename T> Matrix<T>::Matrix(INT _row, INT _col, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil) :
    host(_row, _col, _hdctype & HDCType::Host, _stencil),
    dev(_row, _col, _hdctype & HDCType::Device, _stencil),
    shape(INT2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    name(_name),
    stencil(_stencil)
{
    if (hdctype & HDCType::Device) {
        devptr = (MatrixFrame<T>*)falmMallocDevice(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
    }
}

template<typename T> Matrix<T>::~Matrix() {
    if (hdctype & HDCType::Device) {
        falmFreeDevice(devptr);
    }
    devptr = nullptr;
    hdctype = HDCType::Empty;
}

template<typename T> void Matrix<T>::alloc(INT3 _dom, INT _dim, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil) {
    assert(hdctype == HDCType::Empty);
    host.alloc(_dom, _dim, _hdctype & HDCType::Host, _stencil);
    dev.alloc(_dom, _dim, _hdctype & HDCType::Device, _stencil);
    shape   = INT2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    name    = _name;
    stencil = _stencil;
    if (hdctype & HDCType::Device) {
        devptr = (MatrixFrame<T>*)falmMallocDevice(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
    }
}

template<typename T> void Matrix<T>::alloc(INT _row, INT _col, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil) {
    assert(hdctype == HDCType::Empty);
    host.alloc(_row, _col, _hdctype & HDCType::Host, _stencil);
    dev.alloc(_row, _col, _hdctype & HDCType::Device, _stencil);
    shape   = INT2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    name    = _name;
    stencil = _stencil;
    if (hdctype & HDCType::Device) {
        devptr = (MatrixFrame<T>*)falmMallocDevice(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
    }
}

template<typename T> void Matrix<T>::release(FLAG _hdctype) {
    if (_hdctype & HDCType::Host) {
        assert(hdctype & HDCType::Host);
        host.release();
        hdctype &= ~(HDCType::Host);
    }
    if (_hdctype & HDCType::Device) {
        assert(hdctype & HDCType::Device);
        dev.release();
        falmFreeDevice(devptr);
        devptr = nullptr;
        hdctype &= ~(HDCType::Device);
    }
}

template<typename T> void Matrix<T>::sync(FLAG _mcptype) {
    if (_mcptype == MCpType::Hst2Dev) {
        assert(hdctype & HDCType::Host);
        if (hdctype & HDCType::Device) {
            falmMemcpy(dev.ptr, host.ptr, sizeof(T) * size, MCpType::Hst2Dev);
        } else {
            dev.alloc(shape[0], shape[1], HDCType::Device, stencil);
            falmMemcpy(dev.ptr, host.ptr, sizeof(T) * size, MCpType::Hst2Dev);
            devptr = (MatrixFrame<T>*)falmMallocDevice(sizeof(MatrixFrame<T>));
            falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
            hdctype |= HDCType::Device;
        }
    } else if (_mcptype == MCpType::Dev2Hst) {
        assert(hdctype & HDCType::Device);
        if (hdctype & HDCType::Host) {
            falmMemcpy(host.ptr, dev.ptr, sizeof(T) * size, MCpType::Dev2Hst);
        } else {
            host.alloc(shape[0], shape[1], HDCType::Host, stencil);
            falmMemcpy(host.ptr, dev.ptr, sizeof(T) * size, MCpType::Dev2Hst);
            hdctype |= HDCType::Host;
        }
    }
}

template<typename T> void Matrix<T>::copy(Matrix<T> &src, FLAG _hdctype) {
    if (_hdctype & HDCType::Host) {
        assert((hdctype & src.hdctype & HDCType::Host) && (size == src.size));
        falmMemcpy(host.ptr, src.host.ptr, sizeof(T) * size, MCpType::Hst2Hst);
    }
    if (_hdctype & HDCType::Device) {
        assert((hdctype & src.hdctype & HDCType::Device) && (size == src.size));
        falmMemcpy(dev.ptr, src.dev.ptr, sizeof(T) * size, MCpType::Dev2Dev);
    }
}

template<typename T> void Matrix<T>::clear(FLAG _hdctype) {
    if (_hdctype & HDCType::Host) {
        assert(hdctype & HDCType::Host);
        // falmMemset(host.ptr, 0, sizeof(T) * size);
        host.clear();
    }
    if (_hdctype & HDCType::Device) {
        assert(hdctype & HDCType::Device);
        // falmMemsetDevice(dev.ptr, 0, sizeof(T) * size);
        dev.clear();
    }
}

}

#endif
