#ifndef FALM_CPMBV2_H
#define FALM_CPMBV2_H

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

struct CPMBuffer {
    void            *ptr;
    Mapper           map;
    int             size;
    size_t        dwidth;
    unsigned int buftype;
    unsigned int hdctype;
    unsigned int   color;

    CPMBuffer() : ptr(nullptr), size(0), buftype(BufType::Empty), hdctype(HDCType::Empty) {}
    ~CPMBuffer() {
        if (hdctype == HDCType::Host) {
            falmHostFreePtr(ptr);
        } else if (hdctype == HDCType::Device) {
            falmDevFreePtr(ptr);
        }
    }

    void alloc(size_t _dwidth, uint3 _buf_shape, uint3 _buf_offset, unsigned int _buftype, unsigned int _hdctype) {
        assert(hdctype == HDCType::Empty);
        assert(buftype == BufType::Empty);
        dwidth  = _dwidth;
        map     = Mapper(_buf_shape, _buf_offset);
        size    = PRODUCT3(_buf_shape);
        buftype = _buftype;
        hdctype = _hdctype;
        if (hdctype == HDCType::Host) {
            ptr = falmHostMalloc(dwidth * size);
        } else if (hdctype == HDCType::Device) {
            ptr = falmDevMalloc(dwidth * size);
        }
    }

    void alloc(size_t _dwidth, uint3 _buf_shape, uint3 _buf_offset, unsigned int _buftype, unsigned int _hdctype, Mapper &_pdom, unsigned int _color) {
        assert(hdctype == HDCType::Empty);
        assert(buftype == BufType::Empty);
        dwidth  = _dwidth;
        map     = Mapper(_buf_shape, _buf_offset);
        buftype = _buftype;
        hdctype = _hdctype;
        color   = _color;
        unsigned int refcolor = (SUM3(_pdom.offset) + SUM3(map.offset)) % 2;
        size = map.size / 2;
        if (hdctype == HDCType::Host) {
            ptr = falmHostMalloc(dwidth * size);
        } else if (hdctype == HDCType::Device) {
            ptr = falmDevMalloc(dwidth * size);
        }
    }

    void release() {
        if (hdctype == HDCType::Host) {
            falmHostFreePtr(ptr);
        } else if (hdctype == HDCType::Device) {
            falmDevFreePtr(ptr);
        }
        hdctype = HDCType::Empty;
        buftype = BufType::Empty;
        ptr = nullptr;
    }
};

}

#endif
