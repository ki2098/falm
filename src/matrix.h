#ifndef FALM_MATRIX_H
#define FALM_MATRIX_H

#include <assert.h>
#include <string>
#include "util.h"

namespace Falm {

template<typename T>
struct MatrixFrame {
    T               *ptr;
    INTx2          shape;
    INT             size;
    FLAG         hdctype;
    __host__ __device__ T &operator()(INT _idx) const {return ptr[_idx];}
    __host__ __device__ T &operator()(INT _row, INT _col) const {return ptr[_row + _col * shape.x];}

    MatrixFrame(const MatrixFrame<T> &_mat) = delete;
    MatrixFrame<T>& operator=(const MatrixFrame<T> &_mat) = delete;

    MatrixFrame() : ptr(nullptr), shape(INTx2{0, 0}), size(0), hdctype(HDCType::Empty){}
    MatrixFrame(INTx3 _dom, INT _dim, FLAG _hdctype);
    MatrixFrame(INT _row, INT _col, FLAG _hdctype);
    ~MatrixFrame();

    void alloc(INTx3 _dom, INT _dim, FLAG _hdctype);
    void alloc(INT _row, INT _col, FLAG _hdctype);
    void release();
    void clear() {
        if (hdctype == HDCType::Host) {
            falmMemset(ptr, 0, sizeof(T) * size);
        } else if (hdctype == HDCType::Device) {
            falmMemsetDevice(ptr, 0, sizeof(T) * size);
        }
    }
};

template<typename T> MatrixFrame<T>::MatrixFrame(INTx3 _dom, INT _dim, FLAG _hdctype) :
    shape(INTx2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype)
{
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmMallocPinned(sizeof(T) * size);
        falmMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmMallocDevice(sizeof(T) * size);
        falmMemsetDevice(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> MatrixFrame<T>::MatrixFrame(INT _row, INT _col, FLAG _hdctype) :
    shape(INTx2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype)
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

template<typename T> void MatrixFrame<T>::alloc(INTx3 _dom, INT _dim, FLAG _hdctype) {
    assert(hdctype == HDCType::Empty);
    shape   = INTx2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmMallocPinned(sizeof(T) * size);
        falmMemset(ptr, 0, sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmMallocDevice(sizeof(T) * size);
        falmMemsetDevice(ptr, 0, sizeof(T) * size);
    }
}

template<typename T> void MatrixFrame<T>::alloc(INT _row, INT _col, FLAG _hdctype) {
    assert(hdctype == HDCType::Empty);
    shape   = INTx2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
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

    INTx2            shape;
    INT               size;
    FLAG           hdctype;
    std::string      name;

    __host__ __device__ T &operator()(INT _idx) const {return host(_idx);}
    __host__ __device__ T &operator()(INT _row, INT _col) const {return host(_row, _col);}

    Matrix(const Matrix<T> &_mat) = delete;
    Matrix<T>& operator=(const Matrix<T> &_mat) = delete;

    Matrix(std::string _name = "") : shape(INTx2{0, 0}), size(0), hdctype(HDCType::Empty), name(_name), devptr(nullptr) {}
    Matrix(INTx3 _dom, INT _dim, FLAG _hdctype, std::string _name = "");
    Matrix(INT _row, INT _col, FLAG _hdctype, std::string _name = "");
    ~Matrix();

    void alloc(INTx3 _dom, INT _dim, FLAG _hdctype, std::string _name);
    void alloc(INT _row, INT _col, FLAG _hdctype, std::string _name);
    void alloc(INTx3 _dom, INT _dim, FLAG _hdctype) {
        alloc(_dom, _dim, _hdctype, name);
    }
    void alloc(INT _row, INT _col, FLAG _hdctype) {
        alloc(_row, _col, _hdctype, name);
    }
    void release(FLAG _hdctype);
    void sync(FLAG _mcptype);

    void cpy(Matrix<T> &src, FLAG _hdctype);
    void clear(FLAG _hdctype);

    const char *cname() {return name.c_str();}
};

template<typename T> Matrix<T>::Matrix(INTx3 _dom, INT _dim, FLAG _hdctype, std::string _name) :
    host(_dom, _dim, _hdctype & HDCType::Host),
    dev(_dom, _dim, _hdctype & HDCType::Device),
    shape(INTx2{PRODUCT3(_dom), _dim}),
    size(PRODUCT3(_dom) * _dim),
    hdctype(_hdctype),
    name(_name)
{
    if (hdctype & HDCType::Device) {
        devptr = (MatrixFrame<T>*)falmMallocDevice(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
    }
}

template<typename T> Matrix<T>::Matrix(INT _row, INT _col, FLAG _hdctype, std::string _name) :
    host(_row, _col, _hdctype & HDCType::Host),
    dev(_row, _col, _hdctype & HDCType::Device),
    shape(INTx2{_row, _col}),
    size(_row * _col),
    hdctype(_hdctype),
    name(_name)
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

template<typename T> void Matrix<T>::alloc(INTx3 _dom, INT _dim, FLAG _hdctype, std::string _name) {
    assert(hdctype == HDCType::Empty);
    host.alloc(_dom, _dim, _hdctype & HDCType::Host);
    dev.alloc(_dom, _dim, _hdctype & HDCType::Device);
    shape   = INTx2{PRODUCT3(_dom), _dim};
    size    = PRODUCT3(_dom) * _dim;
    hdctype = _hdctype;
    name    = _name;
    if (hdctype & HDCType::Device) {
        devptr = (MatrixFrame<T>*)falmMallocDevice(sizeof(MatrixFrame<T>));
        falmMemcpy(devptr, &dev, sizeof(MatrixFrame<T>), MCpType::Hst2Dev);
    }
}

template<typename T> void Matrix<T>::alloc(INT _row, INT _col, FLAG _hdctype, std::string _name) {
    assert(hdctype == HDCType::Empty);
    host.alloc(_row, _col, _hdctype & HDCType::Host);
    dev.alloc(_row, _col, _hdctype & HDCType::Device);
    shape   = INTx2{_row, _col};
    size    = _row * _col;
    hdctype = _hdctype;
    name    = _name;
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
            dev.alloc(shape.x, shape.y, HDCType::Device);
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
            host.alloc(shape.x, shape.y, HDCType::Host);
            falmMemcpy(host.ptr, dev.ptr, sizeof(T) * size, MCpType::Dev2Hst);
            hdctype |= HDCType::Host;
        }
    }
}

template<typename T> void Matrix<T>::cpy(Matrix<T> &src, FLAG _hdctype) {
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
