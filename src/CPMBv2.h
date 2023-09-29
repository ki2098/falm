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
    int       count;
    size_t    width;
    FLAG    buftype;
    FLAG    hdctype;
    FLAG      color;

    CPMBuffer() : ptr(nullptr), count(0), buftype(BufType::Empty), hdctype(HDCType::Empty) {}
    ~CPMBuffer() {
        if (hdctype == HDCType::Host) {
            falmFreePinned(ptr);
        } else if (hdctype == HDCType::Device) {
            falmFreeDevice(ptr);
        }
    }

    void alloc(size_t _width, Mapper _map, FLAG _buftype, FLAG _hdctype) {
        assert(hdctype == HDCType::Empty);
        assert(buftype == BufType::Empty);
        width   = _width;
        map     = _map;
        count   = _map.size;
        buftype = _buftype;
        hdctype = _hdctype;
        if (hdctype == HDCType::Host) {
            ptr = falmMallocPinned(width * count);
        } else if (hdctype == HDCType::Device) {
            ptr = falmMallocDevice(width * count);
        }
    }

    void allocColored(size_t _width, Mapper _map, INT _color, FLAG _buftype, FLAG _hdctype, Mapper &_pdm) {
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
            ptr = falmMallocPinned(width * count);
        } else if (hdctype == HDCType::Device) {
            ptr = falmMallocDevice(width * count);
        }
    }
    
    void release() {
        if (hdctype == HDCType::Host) {
            falmFreePinned(ptr);
        } else if (hdctype == HDCType::Device) {
            falmFreeDevice(ptr);
        }
        hdctype = HDCType::Empty;
        buftype = BufType::Empty;
        ptr = nullptr;
    }

    // void alloc(size_t _width, INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype) {
    //     assert(hdctype == HDCType::Empty);
    //     assert(buftype == BufType::Empty);
    //     width   = _width;
    //     map     = Mapper(_buf_shape, _buf_offset);
    //     count   = PRODUCT3(_buf_shape);
    //     buftype = _buftype;
    //     hdctype = _hdctype;
    //     if (hdctype == HDCType::Host) {
    //         ptr = falmMallocPinned(width * count);
    //     } else if (hdctype == HDCType::Device) {
    //         ptr = falmMallocDevice(width * count);
    //     }
    // }

    // void allocAsync(size_t _width, INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype, STREAM stream) {
    //     assert(hdctype == HDCType::Empty);
    //     assert(buftype == BufType::Empty);
    //     assert(_hdctype == HDCType::Device);
    //     width   = _width;
    //     map     = Mapper(_buf_shape, _buf_offset);
    //     count   = PRODUCT3(_buf_shape);
    //     buftype = _buftype;
    //     hdctype = _hdctype;
    //     falmMallocDeviceAsync(&ptr, width * count, stream);
    // }

    // void allocAsync(size_t _width, Mapper _map, FLAG _buftype, FLAG _hdctype, STREAM stream = (STREAM)0) {
    //     assert(hdctype == HDCType::Empty);
    //     assert(buftype == BufType::Empty);
    //     width   = _width;
    //     map     = _map;
    //     count   = _map.size;
    //     buftype = _buftype;
    //     hdctype = _hdctype;
    //     if (hdctype == HDCType::Host) {
    //         ptr = falmMallocPinned(width * count);
    //     } else if (hdctype == HDCType::Device) {
    //         falmMallocDeviceAsync(&ptr, width * count, stream);
    //     }
    // }

    // void alloc(size_t _width, INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype, Mapper &_pdm, INT _color) {
    //     assert(hdctype == HDCType::Empty);
    //     assert(buftype == BufType::Empty);
    //     width   = _width;
    //     map     = Mapper(_buf_shape, _buf_offset);
    //     buftype = _buftype;
    //     hdctype = _hdctype;
    //     color   = _color;
    //     INT refcolor = (SUM3(_pdm.offset) + SUM3(map.offset)) % 2;
    //     count = map.size / 2;
    //     if (map.size % 2 == 1 && refcolor == color) {
    //         count ++;
    //     }
    //     if (hdctype == HDCType::Host) {
    //         ptr = falmMallocPinned(width * count);
    //     } else if (hdctype == HDCType::Device) {
    //         ptr = falmMallocDevice(width * count);
    //     }
    // }

    // void allocAsync(size_t _width, INTx3 _buf_shape, INTx3 _buf_offset, FLAG _buftype, FLAG _hdctype, Mapper &_pdm, INT _color, STREAM stream) {
    //     assert(hdctype == HDCType::Empty);
    //     assert(buftype == BufType::Empty);
    //     assert(_hdctype == HDCType::Device);
    //     width   = _width;
    //     map     = Mapper(_buf_shape, _buf_offset);
    //     buftype = _buftype;
    //     hdctype = _hdctype;
    //     color   = _color;
    //     INT refcolor = (SUM3(_pdm.offset) + SUM3(map.offset)) % 2;
    //     count = map.size / 2;
    //     if (map.size % 2 == 1 && refcolor == color) {
    //         count ++;
    //     }
    //     falmMallocDeviceAsync(&ptr, width * count, stream);
    // }

    // void allocColoredAsync(size_t _width, Mapper _map, FLAG _buftype, FLAG _hdctype, Mapper &_pdm, INT _color, STREAM stream = (STREAM)0) {
    //     assert(hdctype == HDCType::Empty);
    //     assert(buftype == BufType::Empty);
    //     width   = _width;
    //     map     = _map;
    //     buftype = _buftype;
    //     hdctype = _hdctype;
    //     color   = _color;
    //     INT refcolor = (SUM3(_pdm.offset) + SUM3(map.offset)) % 2;
    //     count = map.size / 2;
    //     if (map.size % 2 == 1 && refcolor == color) {
    //         count ++;
    //     }
    //     if (hdctype == HDCType::Host) {
    //         ptr = falmMallocPinned(width * count);
    //     } else if (hdctype == HDCType::Device) {
    //         falmMallocDeviceAsync(&ptr, width * count, stream);
    //     }
    // }

    // void releaseAsync(STREAM stream = (STREAM)0) {
    //     if (hdctype == HDCType::Host) {
    //         falmFreePinned(ptr);
    //     } else if (hdctype == HDCType::Device) {
    //         falmFreeDeviceAsync(ptr, stream);
    //     }
    //     hdctype = HDCType::Empty;
    //     buftype = BufType::Empty;
    //     ptr = nullptr;
    // }
};

}

#endif
