#ifndef FALM_CPMB_H
#define FALM_CPMB_H

#include <stdio.h>
#include <assert.h>
#include "region.h"

namespace Falm {

class CPMBufferType {
public:
    static const FLAG Empty = 0;
    static const FLAG In    = 1;
    static const FLAG Out   = 2;
    static const FLAG InOut = In | Out;
};

typedef CPMBufferType BufType;

template<typename T>
struct CPMBuffer {
    T       *ptr;
    Region   map;
    INT     size;
    FLAG buftype;
    FLAG hdctype;
    INT    color;

    CPMBuffer() : ptr(nullptr), size(0), buftype(BufType::Empty), hdctype(HDCType::Empty) {}
    CPMBuffer(INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype);
    CPMBuffer(INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype, Region &_pdm, INT _color);
    ~CPMBuffer();

    void alloc(INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype);
    void alloc(INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype, Region &_pdm, INT _color);
    void release();

    void clear() {
        if (hdctype == HDCType::Host) {
            falmHostMemset(ptr, 0, sizeof(T) * size);
        } else if (hdctype == HDCType::Device) {
            falmDevMemset(ptr, 0, sizeof(T) * size);
        }
    }
};

template<typename T> CPMBuffer<T>::CPMBuffer(INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype) :
    map(_buf_shape, _buf_offset),
    size(PRODUCT3(_buf_shape)),
    buftype(_buftype),
    hdctype(_hdctype)
{
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMallocPinned(sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
    }
}

template<typename T> CPMBuffer<T>::CPMBuffer(INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype, Region &_pdm, INT _color) :
    map(_buf_shape, _buf_offset),
    buftype(_buftype),
    hdctype(_hdctype),
    color(_color)
{
    INT refcolor = (SUM3(_pdm.offset) + SUM3(map.offset)) % 2;
    size = map.size / 2;
    if (map.size % 2 == 1 && refcolor == color) {
        size ++;
    }
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMallocPinned(sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
    }
}

template<typename T> CPMBuffer<T>::~CPMBuffer() {
    if (hdctype == HDCType::Host) {
        falmHostFreePtr(ptr);
    } else if (hdctype == HDCType::Device) {
        falmDevFreePtr(ptr);
    }
}

template<typename T> void CPMBuffer<T>::alloc(INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype) {
    assert(hdctype == HDCType::Empty);
    assert(buftype == BufType::Empty);
    
    map     = Region(_buf_shape, _buf_offset);
    size    = PRODUCT3(_buf_shape);
    buftype = _buftype;
    hdctype = _hdctype;
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMallocPinned(sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
    }
}

template<typename T> void CPMBuffer<T>::alloc(INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype, Region &_pdm, INT _color) {
    assert(hdctype == HDCType::Empty);
    assert(buftype == BufType::Empty);
    map     = Region(_buf_shape, _buf_offset);
    buftype = _buftype;
    hdctype = _hdctype;
    color   = _color;
    INT refcolor = (SUM3(_pdm.offset) + SUM3(map.offset)) % 2;
    size = map.size / 2;
    if (map.size % 2 == 1 && refcolor == color) {
        size ++;
    }
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMallocPinned(sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
    }
}

template<typename T> void CPMBuffer<T>::release() {
    if (hdctype == HDCType::Host) {
        falmHostFreePtr(ptr);
    } else if (hdctype == HDCType::Device) {
        falmDevFreePtr(ptr);
    }
    hdctype = HDCType::Empty;
    buftype = BufType::Empty;
    ptr = nullptr;
}

}

#endif
