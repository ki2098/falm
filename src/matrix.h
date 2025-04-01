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
    Int2            shape;
    Int              size;
    Flag          hdctype;
    StencilMatrix stencil;
    __host__ __device__ T &operator()(Int _idx) const {return ptr[_idx];}
    __host__ __device__ T &operator()(Int _row, Int _col) const {return ptr[_row + _col * shape[0]];}

    MatrixFrame(const MatrixFrame<T> &_mat) = delete;
    MatrixFrame<T>& operator=(const MatrixFrame<T> &_mat) = delete;

    MatrixFrame() : ptr(nullptr), shape(Int2{{0, 0}}), size(0), hdctype(HDC::Empty), stencil(StencilMatrix::Empty) {}
    MatrixFrame(Int3 _dom, Int _dim, Flag _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    MatrixFrame(Int _row, Int _col, Flag _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    ~MatrixFrame();

    void alloc(Int3 _dom, Int _dim, Flag _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    void alloc(Int _row, Int _col, Flag _hdctype, StencilMatrix _stencil = StencilMatrix::Empty);
    void release();
    void clear() {
        if (hdctype == HDC::Host) {
            falmErrCheckMacro(falmMemset(ptr, 0, sizeof(T) * size));
        } else if (hdctype == HDC::Device) {
            falmErrCheckMacro(falmMemsetDevice(ptr, 0, sizeof(T) * size));
        }
    }
};

template<typename T> MatrixFrame<T>::MatrixFrame(Int3 _dom, Int _dim, Flag _hdctype, StencilMatrix _stencil) :
    shape(Int2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    stencil(_stencil)
{
    // printf("matrix frame constructor called\n");
    if (hdctype == HDC::Host) {
        falmErrCheckMacro(falmMalloc((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemset(ptr, 0, sizeof(T) * size));
    } else if (hdctype == HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemsetDevice(ptr, 0, sizeof(T) * size));
    }
}

template<typename T> MatrixFrame<T>::MatrixFrame(Int _row, Int _col, Flag _hdctype, StencilMatrix _stencil) :
    shape(Int2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    stencil(_stencil)
{
    // printf("matrix frame constructor called\n");
    if (hdctype == HDC::Host) {
        falmErrCheckMacro(falmMalloc((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemset(ptr, 0, sizeof(T) * size));
    } else if (hdctype == HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&ptr, sizeof(T) * size));
        falmErrCheckMacro(falmMemsetDevice(ptr, 0, sizeof(T) * size));
    }
}

template<typename T> MatrixFrame<T>::~MatrixFrame() {
    // printf("matrix frame destructor called\n");
    if (hdctype == HDC::Host) {
        falmErrCheckMacro(falmFree(ptr));
    } else if (hdctype == HDC::Device) {
        falmErrCheckMacro(falmFreeDevice(ptr));
    }
    ptr = nullptr;
    hdctype = HDC::Empty;
}

template<typename T> void MatrixFrame<T>::alloc(Int3 _dom, Int _dim, Flag _hdctype, StencilMatrix _stencil) {
    // printf("matrix frame allocator called\n");
    assert(hdctype == HDC::Empty);
    shape   = Int2{PRODUCT3(_dom), _dim};
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

template<typename T> void MatrixFrame<T>::alloc(Int _row, Int _col, Flag _hdctype, StencilMatrix _stencil) {
    // printf("matrix frame allocator called\n");
    assert(hdctype == HDC::Empty);
    shape   = Int2{_row, _col};
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
    // printf("matrix frame release called\n");
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

    Int2             shape;
    Int               size;
    Flag           hdctype;
    std::string       name;
    StencilMatrix    stencil;

    __host__ __device__ T &operator()(Int _idx) const {return host(_idx);}
    __host__ __device__ T &operator()(Int _row, Int _col) const {return host(_row, _col);}

    Matrix(const Matrix<T> &_mat) = delete;
    Matrix<T>& operator=(const Matrix<T> &_mat) = delete;

    Matrix(std::string _name = "") : shape(Int2{{0, 0}}), size(0), hdctype(HDC::Empty), name(_name), devptr(nullptr), stencil(StencilMatrix::Empty) {}
    Matrix(Int3 _dom, Int _dim, Flag _hdctype, const std::string &_name, StencilMatrix _stencil = StencilMatrix::Empty);
    Matrix(Int _row, Int _col, Flag _hdctype, const std::string &_name, StencilMatrix _stencil = StencilMatrix::Empty);
    Matrix(Int3 _dom, Int _dim, Flag _hdctype, StencilMatrix _stencil = StencilMatrix::Empty) : Matrix(_dom, _dim, _hdctype, "", _stencil) {}
    Matrix(Int _row, Int _col, Flag _hdctype, StencilMatrix _stencil = StencilMatrix::Empty) : Matrix(_row, _col, _hdctype, "", _stencil) {}
    
    ~Matrix();

    void alloc(Int3 _dom, Int _dim, Flag _hdctype, const std::string &_name, StencilMatrix _stencil = StencilMatrix::Empty);
    void alloc(Int _row, Int _col, Flag _hdctype, const std::string &_name, StencilMatrix _stencil = StencilMatrix::Empty);
    void alloc(Int3 _dom, Int _dim, Flag _hdctype, StencilMatrix _stencil = StencilMatrix::Empty) {
        alloc(_dom, _dim, _hdctype, name, _stencil);
    }
    void alloc(Int _row, Int _col, Flag _hdctype, StencilMatrix _stencil = StencilMatrix::Empty) {
        alloc(_row, _col, _hdctype, name, _stencil);
    }
    void release(Flag _hdctype);
    void release();
    void sync(Flag _mcptype);

    void copy(Matrix<T> &src, Flag _hdctype);
    void clear(Flag _hdctype);

    const char *cname() {return name.c_str();}
};

template<typename T> Matrix<T>::Matrix(Int3 _dom, Int _dim, Flag _hdctype, const std::string &_name, StencilMatrix _stencil) :
    host(_dom, _dim, _hdctype & HDC::Host, _stencil),
    dev(_dom, _dim, _hdctype & HDC::Device, _stencil),
    shape(Int2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    name(_name),
    stencil(_stencil)
{
    // printf("matrix constructor called\n");
    if (hdctype & HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(MatrixFrame<T>)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCP::Hst2Dev));
    }
}

template<typename T> Matrix<T>::Matrix(Int _row, Int _col, Flag _hdctype, const std::string &_name, StencilMatrix _stencil) :
    host(_row, _col, _hdctype & HDC::Host, _stencil),
    dev(_row, _col, _hdctype & HDC::Device, _stencil),
    shape(Int2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    name(_name),
    stencil(_stencil)
{
    // printf("matrix constructor called\n");
    if (hdctype & HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(MatrixFrame<T>)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCP::Hst2Dev));
    }
}

template<typename T> Matrix<T>::~Matrix() {
    // printf("matrix destructor called\n");
    // printf("desctuctor for matrix %s\n", name.c_str());
    if (hdctype & HDC::Device) {
        falmErrCheckMacro(falmFreeDevice(devptr));
    }
}

template<typename T> void Matrix<T>::alloc(Int3 _dom, Int _dim, Flag _hdctype, const std::string &_name, StencilMatrix _stencil) {
    // printf("matrix allocator called\n");
    assert(hdctype == HDC::Empty);
    host.alloc(_dom, _dim, _hdctype & HDC::Host, _stencil);
    dev.alloc(_dom, _dim, _hdctype & HDC::Device, _stencil);
    shape   = Int2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    name    = _name;
    stencil = _stencil;
    if (hdctype & HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(MatrixFrame<T>)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCP::Hst2Dev));
    }
}

template<typename T> void Matrix<T>::alloc(Int _row, Int _col, Flag _hdctype, const std::string &_name, StencilMatrix _stencil) {
    // printf("matrix destructor called\n");
    assert(hdctype == HDC::Empty);
    host.alloc(_row, _col, _hdctype & HDC::Host, _stencil);
    dev.alloc(_row, _col, _hdctype & HDC::Device, _stencil);
    shape   = Int2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    name    = _name;
    stencil = _stencil;
    if (hdctype & HDC::Device) {
        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(MatrixFrame<T>)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCP::Hst2Dev));
    }
}

template<typename T> void Matrix<T>::release(Flag _hdctype) {
    // printf("matrix release called\n");
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
    // printf("matrix release called\n");
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

template<typename T> void Matrix<T>::sync(Flag _mcptype) {
    // printf("matrix sync called\n");
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

template<typename T> void Matrix<T>::copy(Matrix<T> &src, Flag _hdctype) {
    // printf("matrix copy called\n");
    if (_hdctype & hdctype & HDC::Host) {
        assert((src.hdctype & HDC::Host) && (size == src.size));
        falmErrCheckMacro(falmMemcpy(host.ptr, src.host.ptr, sizeof(T) * size, MCP::Hst2Hst));
    }
    if (_hdctype & hdctype & HDC::Device) {
        assert((src.hdctype & HDC::Device) && (size == src.size));
        falmErrCheckMacro(falmMemcpy(dev.ptr, src.dev.ptr, sizeof(T) * size, MCP::Dev2Dev));
    }
}

template<typename T> void Matrix<T>::clear(Flag _hdctype) {
    // printf("matrix clear called\n");
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
