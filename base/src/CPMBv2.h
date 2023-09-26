#ifndef FALM_CPMBV2_H
#define FALM_CPMBV2_H

#include <stdio.h>
#include <assert.h>
#include "mapper.h"

namespace Falm {

class CPMBufferType {
public:
    static const FLAG Empty = 0;
    static const FLAG In    = 1;
    static const FLAG Out   = 2;
    static const FLAG InOut = In | Out;
};

typedef CPMBufferType BufType;

struct CPMBuffer {
    void       *ptr;
    Mapper      map;
    INT       count;
    INT       width;
    FLAG    buftype;
    FLAG    hdctype;
    FLAG      color;

    CPMBuffer() : ptr(nullptr), count(0), buftype(BufType::Empty), hdctype(HDCType::Empty) {}
    ~CPMBuffer() {
        if (hdctype == HDCType::Host) {
            falmHostFreePtr(ptr);
        } else if (hdctype == HDCType::Device) {
            falmDevFreePtr(ptr);
        }
    }

    void alloc(INT _width, INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype) {
        assert(hdctype == HDCType::Empty);
        assert(buftype == BufType::Empty);
        width   = _width;
        map     = Mapper(_buf_shape, _buf_offset);
        count   = PRODUCT3(_buf_shape);
        buftype = _buftype;
        hdctype = _hdctype;
        if (hdctype == HDCType::Host) {
            ptr = falmHostMalloc(width * count);
        } else if (hdctype == HDCType::Device) {
            ptr = falmDevMalloc(width * count);
        }
    }

    void alloc(INT _width, INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype, Mapper &_pdom, INT _color) {
        assert(hdctype == HDCType::Empty);
        assert(buftype == BufType::Empty);
        width   = _width;
        map     = Mapper(_buf_shape, _buf_offset);
        buftype = _buftype;
        hdctype = _hdctype;
        color   = _color;
        INT refcolor = (SUM3(_pdom.offset) + SUM3(map.offset)) % 2;
        count = map.size / 2;
        if (map.size % 2 == 1 && refcolor == color) {
            count ++;
        }
        if (hdctype == HDCType::Host) {
            ptr = falmHostMalloc(width * count);
        } else if (hdctype == HDCType::Device) {
            ptr = falmDevMalloc(width * count);
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
