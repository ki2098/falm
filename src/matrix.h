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

    MatrixFrame() : ptr(nullptr), shape(INT2{{0, 0}}), size(0), hdctype(HDC::Empty), stencil(StencilMatrix::Empty) {}
    MatrixFrame(INT3 _dom, INT _dim, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    MatrixFrame(INT _row, INT _col, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    ~MatrixFrame();

    void alloc(INT3 _dom, INT _dim, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    void alloc(INT _row, INT _col, FLAG _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    void release();
    void clear() {
        if (hdctype == HDC::Host) {
            falmErrCheckMacro(falmMemset(ptr, 0, sizeof(T) * size));
        } else if (hdctype == HDC::Device) {
            falmErrCheckMacro(falmMemsetDevice(ptr, 0, sizeof(T) * size));
        }
    }
};

template<typename T> MatrixFrame<T>::MatrixFrame(INT3 _dom, INT _dim, FLAG _hdctype, StencilMatrix _stencil) :
    shape(INT2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    stencil(_stencil)
{
    if (hdctype == HDC::Host) {
        falmErrCheckMacro(falmMalloc((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemset(ptr, 0, sizeof(T) * size));
    } else if (hdctype == HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemsetDevice(ptr, 0, sizeof(T) * size));
    }
}

template<typename T> MatrixFrame<T>::MatrixFrame(INT _row, INT _col, FLAG _hdctype, StencilMatrix _stencil) :
    shape(INT2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    stencil(_stencil)
{
    if (hdctype == HDC::Host) {
        falmErrCheckMacro(falmMalloc((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemset(ptr, 0, sizeof(T) * size));
    } else if (hdctype == HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemsetDevice(ptr, 0, sizeof(T) * size));
    }
}

template<typename T> MatrixFrame<T>::~MatrixFrame() {
    if (hdctype == HDC::Host) {
        falmErrCheckMacro(falmFree(ptr));
    } else if (hdctype == HDC::Device) {
        falmErrCheckMacro(falmFreeDevice(ptr));
    }
    ptr = nullptr;
    hdctype = HDC::Empty;
}

template<typename T> void MatrixFrame<T>::alloc(INT3 _dom, INT _dim, FLAG _hdctype, StencilMatrix _stencil) {
    assert(hdctype == HDC::Empty);
    shape   = INT2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    stencil = _stencil;
    if (hdctype == HDC::Host) {
        falmErrCheckMacro(falmMalloc((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemset(ptr, 0, sizeof(T) * size));
    } else if (hdctype == HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemsetDevice(ptr, 0, sizeof(T) * size));
    }
}

template<typename T> void MatrixFrame<T>::alloc(INT _row, INT _col, FLAG _hdctype, StencilMatrix _stencil) {
    assert(hdctype == HDC::Empty);
    shape   = INT2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    stencil = _stencil;
    if (hdctype == HDC::Host) {
        falmErrCheckMacro(falmMalloc((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemset(ptr, 0, sizeof(T) * size));
    } else if (hdctype == HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemsetDevice(ptr, 0, sizeof(T) * size));
    }
}

template<typename T> void MatrixFrame<T>::release() {
    if (hdctype == HDC::Host) {
        falmErrCheckMacro(falmFree(ptr));
    } else if (hdctype == HDC::Device) {
        falmErrCheckMacro(falmFreeDevice(ptr));
    }
    ptr = nullptr;
    hdctype = HDC::Empty;
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

    Matrix(std::string _name = "") : shape(INT2{{0, 0}}), size(0), hdctype(HDC::Empty), name(_name), devptr(nullptr), stencil(StencilMatrix::Empty) {}
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
    void release();
    void sync(FLAG _mcptype);

    void copy(Matrix<T> &src, FLAG _hdctype);
    void clear(FLAG _hdctype);

    const char *cname() {return name.c_str();}
};

template<typename T> Matrix<T>::Matrix(INT3 _dom, INT _dim, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil) :
    host(_dom, _dim, _hdctype & HDC::Host, _stencil),
    dev(_dom, _dim, _hdctype & HDC::Device, _stencil),
    shape(INT2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    name(_name),
    stencil(_stencil)
{
    if (hdctype & HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(MatrixFrame<T>)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCP::Hst2Dev));
    }
}

template<typename T> Matrix<T>::Matrix(INT _row, INT _col, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil) :
    host(_row, _col, _hdctype & HDC::Host, _stencil),
    dev(_row, _col, _hdctype & HDC::Device, _stencil),
    shape(INT2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    name(_name),
    stencil(_stencil)
{
    if (hdctype & HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(MatrixFrame<T>)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCP::Hst2Dev));
    }
}

template<typename T> Matrix<T>::~Matrix() {
    // printf("desctuctor for matrix %s\n", name.c_str());
    if (hdctype & HDC::Device) {
        falmErrCheckMacro(falmFreeDevice(devptr));
    }
}

template<typename T> void Matrix<T>::alloc(INT3 _dom, INT _dim, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil) {
    assert(hdctype == HDC::Empty);
    host.alloc(_dom, _dim, _hdctype & HDC::Host, _stencil);
    dev.alloc(_dom, _dim, _hdctype & HDC::Device, _stencil);
    shape   = INT2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    name    = _name;
    stencil = _stencil;
    if (hdctype & HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(MatrixFrame<T>)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCP::Hst2Dev));
    }
}

template<typename T> void Matrix<T>::alloc(INT _row, INT _col, FLAG _hdctype, const std::string &_name, StencilMatrix _stencil) {
    assert(hdctype == HDC::Empty);
    host.alloc(_row, _col, _hdctype & HDC::Host, _stencil);
    dev.alloc(_row, _col, _hdctype & HDC::Device, _stencil);
    shape   = INT2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    name    = _name;
    stencil = _stencil;
    if (hdctype & HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(MatrixFrame<T>)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCP::Hst2Dev));
    }
}

template<typename T> void Matrix<T>::release(FLAG _hdctype) {
    if (_hdctype & hdctype & HDC::Host) {
        // assert(hdctype & HDCType::Host);
        host.release();
        hdctype &= ~(HDC::Host);
    }
    if (_hdctype & hdctype & HDC::Device) {
        // assert(hdctype & HDCType::Device);
        dev.release();
        falmErrCheckMacro(falmFreeDevice(devptr));
        devptr = nullptr;
        hdctype &= ~(HDC::Device);
    }
}

template<typename T> void Matrix<T>::release() {
    if (hdctype & HDC::Host) {
        // assert(hdctype & HDCType::Host);
        host.release();
    }
    if (hdctype & HDC::Device) {
        // assert(hdctype & HDCType::Device);
        dev.release();
        falmErrCheckMacro(falmFreeDevice(devptr));
        devptr = nullptr;
    }
    hdctype = HDC::Empty;
}

template<typename T> void Matrix<T>::sync(FLAG _mcptype) {
    if (_mcptype == MCP::Hst2Dev) {
        assert(hdctype & HDC::Host);
        if (hdctype & HDC::Device) {
            falmErrCheckMacro(falmMemcpy(dev.ptr, host.ptr, sizeof(T) * size, MCP::Hst2Dev));
        } else {
            dev.alloc(shape[0], shape[1], HDC::Device, stencil);
            falmErrCheckMacro(falmMemcpy(dev.ptr, host.ptr, sizeof(T) * size, MCP::Hst2Dev));
            falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(MatrixFrame<T>)));
            falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCP::Hst2Dev));
            hdctype |= HDC::Device;
        }
    } else if (_mcptype == MCP::Dev2Hst) {
        assert(hdctype & HDC::Device);
        if (hdctype & HDC::Host) {
            falmErrCheckMacro(falmMemcpy(host.ptr, dev.ptr, sizeof(T) * size, MCP::Dev2Hst));
        } else {
            host.alloc(shape[0], shape[1], HDC::Host, stencil);
            falmErrCheckMacro(falmMemcpy(host.ptr, dev.ptr, sizeof(T) * size, MCP::Dev2Hst));
            hdctype |= HDC::Host;
        }
    }
    // falmWaitStream();
}

template<typename T> void Matrix<T>::copy(Matrix<T> &src, FLAG _hdctype) {
    if (_hdctype & hdctype & HDC::Host) {
        assert((src.hdctype & HDC::Host) && (size == src.size));
        falmErrCheckMacro(falmMemcpy(host.ptr, src.host.ptr, sizeof(T) * size, MCP::Hst2Hst));
    }
    if (_hdctype & hdctype & HDC::Device) {
        assert((src.hdctype & HDC::Device) && (size == src.size));
        falmErrCheckMacro(falmMemcpy(dev.ptr, src.dev.ptr, sizeof(T) * size, MCP::Dev2Dev));
    }
}

template<typename T> void Matrix<T>::clear(FLAG _hdctype) {
    if (_hdctype & hdctype & HDC::Host) {
        // assert(hdctype & HDC::Host);
        // falmMemset(host.ptr, 0, sizeof(T) * size);
        host.clear();
    }
    if (_hdctype & hdctype & HDC::Device) {
        // assert(hdctype & HDC::Device);
        // falmMemsetDevice(dev.ptr, 0, sizeof(T) * size);
        dev.clear();
    }
}

}

#endif
