#ifndef FALM_CPMBV2_H
#define FALM_CPMBV2_H

#include <stdio.h>
#include <assert.h>
#include "region.h"

namespace Falm {

class CPMBufferType {
public:
    static const Flag Empty = 0;
    static const Flag In    = 1;
    static const Flag Out   = 2;
    static const Flag InOut = In | Out;
};

typedef CPMBufferType BufType;

struct CPMBuffer {
    void       *ptr;
    Region      map;
    int       count;
    size_t    width;
    Flag    buftype;
    Flag    hdctype;
    Int       color;

    CPMBuffer() : ptr(nullptr), count(0), buftype(BufType::Empty), hdctype(HDC::Empty) {}
    ~CPMBuffer() {
        if (hdctype == HDC::Host) {
            falmErrCheckMacro(falmFree(ptr));
        } else if (hdctype == HDC::Device) {
            falmErrCheckMacro(falmFreeDevice(ptr));
        }
    }

    void alloc(size_t _width, const Region &_map, Flag _buftype, Flag _hdctype) {
        assert(hdctype == HDC::Empty);
        assert(buftype == BufType::Empty);
        width   = _width;
        map     = _map;
        count   = _map.size;
        buftype = _buftype;
        hdctype = _hdctype;
        if (hdctype == HDC::Host) {
            falmErrCheckMacro(falmMalloc((void**)&ptr, width * count));
        } else if (hdctype == HDC::Device) {
            falmErrCheckMacro(falmMallocDevice((void**)&ptr, width * count));
        }
    }

    void allocColored(size_t _width, const Region &_map, Int _color, Flag _buftype, Flag _hdctype, Region &_pdm) {
        assert(hdctype == HDC::Empty);
        assert(buftype == BufType::Empty);
        width   = _width;
        map     = _map;
        buftype = _buftype;
        hdctype = _hdctype;
        color   = _color;
        Int refcolor = (SUM3(_pdm.offset) + SUM3(map.offset)) % 2;
        count = map.size / 2;
        if (map.size % 2 == 1 && refcolor == color) {
            count ++;
        }
        if (hdctype == HDC::Host) {
            falmErrCheckMacro(falmMalloc((void**)&ptr, width * count));
        } else if (hdctype == HDC::Device) {
            falmErrCheckMacro(falmMallocDevice((void**)&ptr, width * count));
        }
    }
    
    void release() {
        if (hdctype == HDC::Host) {
            falmErrCheckMacro(falmFree(ptr));
        } else if (hdctype == HDC::Device) {
            falmErrCheckMacro(falmFreeDevice(ptr));
        }
        hdctype = HDC::Empty;
        buftype = BufType::Empty;
        ptr = nullptr;
    }

};

}

#endif

