#ifndef FALM_CPMB_H
#define FALM_CPMB_H

#include <stdio.h>
#include <assert.h>
#include "mapper.h"

namespace Falm {

class CPMBufferType {
public:
    static const UINT Empty = 0;
    static const UINT In    = 1;
    static const UINT Out   = 2;
    static const UINT InOut = In | Out;
};

typedef CPMBufferType BufType;

template<typename T>
struct CPMBuffer {
    T               *ptr;
    Mapper           map;
    UINT    size;
    UINT buftype;
    UINT hdctype;
    UINT   color;

    CPMBuffer() : ptr(nullptr), size(0), buftype(BufType::Empty), hdctype(HDCType::Empty) {}
    CPMBuffer(uint3 _buf_shape, uint3 _buf_offset, UINT _buftype, UINT _hdctype);
    CPMBuffer(uint3 _buf_shape, uint3 _buf_offset, UINT _buftype, UINT _hdctype, Mapper &_pdom, UINT _color);
    ~CPMBuffer();

    void alloc(uint3 _buf_shape, uint3 _buf_offset, UINT _buftype, UINT _hdctype);
    void alloc(uint3 _buf_shape, uint3 _buf_offset, UINT _buftype, UINT _hdctype, Mapper &_pdom, UINT _color);
    void release();

    void clear() {
        if (hdctype == HDCType::Host) {
            falmHostMemset(ptr, 0, sizeof(T) * size);
        } else if (hdctype == HDCType::Device) {
            falmDevMemset(ptr, 0, sizeof(T) * size);
        }
    }
};

template<typename T> CPMBuffer<T>::CPMBuffer(uint3 _buf_shape, uint3 _buf_offset, UINT _buftype, UINT _hdctype) :
    map(_buf_shape, _buf_offset),
    size(PRODUCT3(_buf_shape)),
    buftype(_buftype),
    hdctype(_hdctype)
{
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
    }
}

template<typename T> CPMBuffer<T>::CPMBuffer(uint3 _buf_shape, uint3 _buf_offset, UINT _buftype, UINT _hdctype, Mapper &_pdom, UINT _color) :
    map(_buf_shape, _buf_offset),
    buftype(_buftype),
    hdctype(_hdctype),
    color(_color)
{
    UINT refcolor = (SUM3(_pdom.offset) + SUM3(map.offset)) % 2;
    size = map.size / 2;
    if (map.size % 2 == 1 && refcolor == color) {
        size ++;
    }
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
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

template<typename T> void CPMBuffer<T>::alloc(uint3 _buf_shape, uint3 _buf_offset, UINT _buftype, UINT _hdctype) {
    assert(hdctype == HDCType::Empty);
    assert(buftype == BufType::Empty);
    
    map     = Mapper(_buf_shape, _buf_offset);
    size    = PRODUCT3(_buf_shape);
    buftype = _buftype;
    hdctype = _hdctype;
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
    } else if (hdctype == HDCType::Device) {
        ptr = (T*)falmDevMalloc(sizeof(T) * size);
    }
}

template<typename T> void CPMBuffer<T>::alloc(uint3 _buf_shape, uint3 _buf_offset, UINT _buftype, UINT _hdctype, Mapper &_pdom, UINT _color) {
    assert(hdctype == HDCType::Empty);
    assert(buftype == BufType::Empty);
    map     = Mapper(_buf_shape, _buf_offset);
    buftype = _buftype;
    hdctype = _hdctype;
    color   = _color;
    UINT refcolor = (SUM3(_pdom.offset) + SUM3(map.offset)) % 2;
    size = map.size / 2;
    if (map.size % 2 == 1 && refcolor == color) {
        size ++;
    }
    if (hdctype == HDCType::Host) {
        ptr = (T*)falmHostMalloc(sizeof(T) * size);
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
