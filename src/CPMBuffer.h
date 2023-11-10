#ifndef FALM_CPMBV2_H
#define FALM_CPMBV2_H

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

struct CPMBuffer {
    void       *ptr;
    Region      map;
    int       count;
    size_t    width;
    FLAG    buftype;
    FLAG    hdctype;
    INT       color;

    CPMBuffer() : ptr(nullptr), count(0), buftype(BufType::Empty), hdctype(HDCType::Empty) {}
    ~CPMBuffer() {
        if (hdctype == HDCType::Host) {
            falmFree(ptr);
        } else if (hdctype == HDCType::Device) {
            falmFreeDevice(ptr);
        }
    }

    void alloc(size_t _width, const Region &_map, FLAG _buftype, FLAG _hdctype) {
        assert(hdctype == HDCType::Empty);
        assert(buftype == BufType::Empty);
        width   = _width;
        map     = _map;
        count   = _map.size;
        buftype = _buftype;
        hdctype = _hdctype;
        if (hdctype == HDCType::Host) {
            ptr = falmMalloc(width * count);
        } else if (hdctype == HDCType::Device) {
            ptr = falmMallocDevice(width * count);
        }
    }

    void allocColored(size_t _width, const Region &_map, INT _color, FLAG _buftype, FLAG _hdctype, Region &_pdm) {
        assert(hdctype == HDCType::Empty);
        assert(buftype == BufType::Empty);
        width   = _width;
        map     = _map;
        buftype = _buftype;
        hdctype = _hdctype;
        color   = _color;
        INT refcolor = (SUM3(_pdm.offset) + SUM3(map.offset)) % 2;
        count = map.size / 2;
        if (map.size % 2 == 1 && refcolor == color) {
            count ++;
        }
        if (hdctype == HDCType::Host) {
            ptr = falmMalloc(width * count);
        } else if (hdctype == HDCType::Device) {
            ptr = falmMallocDevice(width * count);
        }
    }
    
    void release() {
        if (hdctype == HDCType::Host) {
            falmFree(ptr);
        } else if (hdctype == HDCType::Device) {
            falmFreeDevice(ptr);
        }
        hdctype = HDCType::Empty;
        buftype = BufType::Empty;
        ptr = nullptr;
    }

};

}

#endif
