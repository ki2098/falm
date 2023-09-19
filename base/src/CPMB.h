#ifndef FALM_CPMB_H
#define FALM_CPMB_H

#include <stdio.h>
#include <assert.h>
#include "mapper.h"

namespace Falm {

class CPMBufferType {
public:
    static const unsigned int Empty = 0;
    static const unsigned int In    = 1;
    static const unsigned int Out   = 2;
    static const unsigned int InOut = In | Out;
};

typedef CPMBufferType BufType;

template<typename T>
struct CPMBuffer {
    T               *ptr;
    Mapper           map;
    unsigned int    size;
    unsigned int buftype;
    unsigned int hdctype;
    unsigned int   color;

    CPMBuffer() : ptr(nullptr), size(0), buftype(BufType::Empty), hdctype(HDCType::Empty) {}
    CPMBuffer(uint3 _buf_shape, uint3 _buf_offset, unsigned int _buftype, unsigned int _hdctype);
    CPMBuffer(uint3 _buf_shape, uint3 _buf_offset, unsigned int _buftype, unsigned int _hdctype, Mapper &_pdom, unsigned int _color);
    ~CPMBuffer();

    void init(uint3 _buf_shape, uint3 _buf_offset, unsigned int _buftype, unsigned int _hdctype);
    void init(uint3 _buf_shape, uint3 _buf_offset, unsigned int _buftype, unsigned int _hdctype, Mapper &_pdom, unsigned int _color);
    void release();

    void clear() {
        if (hdctype == HDCType::Host) {
            falmHostMemset(ptr, 0, sizeof(T) * size);
        } else if (hdctype == HDCType::Device) {
            falmDevMemset(ptr, 0, sizeof(T) * size);
        }
    }
};

template<typename T> CPMBuffer<T>::CPMBuffer(uint3 _buf_shape, uint3 _buf_offset, unsigned int _buftype, unsigned int _hdctype) :
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

template<typename T> CPMBuffer<T>::CPMBuffer(uint3 _buf_shape, uint3 _buf_offset, unsigned int _buftype, unsigned int _hdctype, Mapper &_pdom, unsigned int _color) :
    map(_buf_shape, _buf_offset),
    buftype(_buftype),
    hdctype(_hdctype),
    color(_color)
{
    unsigned int refcolor = (SUM3(_pdom.offset) + SUM3(map.offset)) % 2;
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

template<typename T> void CPMBuffer<T>::init(uint3 _buf_shape, uint3 _buf_offset, unsigned int _buftype, unsigned int _hdctype) {
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

template<typename T> void CPMBuffer<T>::init(uint3 _buf_shape, uint3 _buf_offset, unsigned int _buftype, unsigned int _hdctype, Mapper &_pdom, unsigned int _color) {
    assert(hdctype == HDCType::Empty);
    assert(buftype == BufType::Empty);
    map     = Mapper(_buf_shape, _buf_offset);
    buftype = _buftype;
    hdctype = _hdctype;
    color   = _color;
    unsigned int refcolor = (SUM3(_pdom.offset) + SUM3(map.offset)) % 2;
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