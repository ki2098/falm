#include "region.h"
#include "devdefine.h"
#include "error.h"

namespace Falm {

class CpmBuffer {
public:
    void *ptr = nullptr;
    void *packer = nullptr;
    bool active = false;
    Region map;
    int count = 0;
    size_t capacity = 0;
    size_t width = 0;
    INT color;
    FLAG hdc;
};

class CpmBufMan {
public:
    static const int NBUFFER = 108;
public:
    int max_used_idx = -1;
    CpmBuffer buffer[NBUFFER];
    FLAG hdc;

    CpmBuffer &get(int i) {
        return buffer[i];
    }

    int release(int i) {
        CpmBuffer &buf = buffer[i];
        if (buf.active) {
            return FalmErr::cpmBufReleaseErr;
        }
        // if (buf.hdc & HDC::Device) {
        //     if (falmErrCheckMacro(falmFreeDevice(buf.dptr))) {
        //         return FalmErr::cpmBufReleaseErr;
        //     }
        // }
        // if (buf.hdc & HDC::Host) {
        //     if (falmErrCheckMacro(falmFreePinned(buf.hptr))) {
        //         return FalmErr::cpmBufReleaseErr;
        //     }
        // }
        if (hdc == HDC::Host) {
            falmErrCheckMacro(falmFreeDevice(buf.packer));
            falmErrCheckMacro(falmFreePinned(buf.ptr));
        } else if (hdc == HDC::Device) {
            falmErrCheckMacro(falmFreeDevice(buf.ptr));
        }
        buf.capacity = 0;
        buf.hdc = HDC::Empty;
        return FalmErr::success;
    }

    int request(size_t width, const Region &map, int *buf_id) {
        int count = map.size;
        size_t reqsize = width * count;

        int first_vacant = -1;
        for (int i = 0; i <= max_used_idx; i ++) {
            if (buffer[i].active == false) {
                if (first_vacant == -1) {
                    first_vacant = i;
                }
                if (buffer[i].capacity >= reqsize) {
                    CpmBuffer &buf = buffer[i];
                    buf.active = true;
                    buf.map = map;
                    buf.count = count;
                    buf.width = width;
                    *buf_id = i;
                    return FalmErr::success;
                }
            }
        }
        if (first_vacant == -1) {
            first_vacant = max_used_idx + 1;
            if (first_vacant >= NBUFFER) {
                *buf_id = -1;
                return FalmErr::cpmNoVacantBuffer;
            }

            CpmBuffer &buf = buffer[first_vacant];
            if (hdc == HDC::Host) {
                falmErrCheckMacro(falmMallocDevice(&buf.packer, reqsize));
                falmErrCheckMacro(falmMallocPinned(&buf.ptr, reqsize));
            } else if (hdc == HDC::Device) {
                falmErrCheckMacro(falmMallocDevice(&buf.ptr, reqsize));
            }
            printf("buffer %d -> %lu\n", first_vacant, reqsize);
            buf.hdc = hdc;
            buf.capacity = reqsize;
            buf.map = map;
            buf.count = count;
            buf.active = true;
            buf.width = width;
            *buf_id = first_vacant;
            max_used_idx = first_vacant;
            return FalmErr::success;
        } else {
            if(release(first_vacant)) {
                *buf_id = -1;
                return FalmErr::cpmBufReqErr;
            }

            CpmBuffer &buf = buffer[first_vacant];
            if (hdc == HDC::Host) {
                falmErrCheckMacro(falmMallocDevice(&buf.packer, reqsize));
                falmErrCheckMacro(falmMallocPinned(&buf.ptr, reqsize));
            } else if (hdc == HDC::Device) {
                falmErrCheckMacro(falmMallocDevice(&buf.ptr, reqsize));
            }
            printf("buffer %d -> %lu\n", first_vacant, reqsize);
            buf.hdc = hdc;
            buf.capacity = reqsize;
            buf.map = map;
            buf.count = count;
            buf.active = true;
            buf.width = width;
            *buf_id = first_vacant;
            max_used_idx = first_vacant;
            return FalmErr::success;
        }
    }

    int request(size_t width, const Region &map, const Region &pdm, INT color, int *buf_id) {
        INT refcolor = (SUM3(pdm.offset) + SUM3(map.offset)) % 2;
        int count = map.size / 2;
        if (map.size % 2 == 1 && refcolor == color) {
            count ++;
        }
        size_t reqsize = width * count;
        
        int first_vacant = -1;
        for (int i = 0; i <= max_used_idx; i ++) {
            if (buffer[i].active == false) {
                if (first_vacant == -1) {
                    first_vacant = i;
                }
                if (buffer[i].capacity >= reqsize) {
                    CpmBuffer &buf = buffer[i];
                    buf.active = true;
                    buf.map = map;
                    buf.count = count;
                    buf.width = width;
                    buf.color = color;
                    *buf_id = i;
                    return FalmErr::success;
                }
            }
        }
        if (first_vacant == -1) {
            first_vacant = max_used_idx + 1;
            if (first_vacant >= NBUFFER) {
                *buf_id = -1;
                return FalmErr::cpmNoVacantBuffer;
            }

            CpmBuffer &buf = buffer[first_vacant];
            if (hdc == HDC::Host) {
                falmErrCheckMacro(falmMallocDevice(&buf.packer, reqsize));
                falmErrCheckMacro(falmMallocPinned(&buf.ptr, reqsize));
            } else if (hdc == HDC::Device) {
                falmErrCheckMacro(falmMallocDevice(&buf.ptr, reqsize));
            }
            printf("buffer %d -> %lu\n", first_vacant, reqsize);
            buf.hdc = hdc;
            buf.capacity = reqsize;
            buf.map = map;
            buf.count = count;
            buf.color = color;
            buf.active = true;
            buf.width = width;
            *buf_id = first_vacant;
            max_used_idx = first_vacant;
            return FalmErr::success;
        } else {
            if(release(first_vacant)) {
                *buf_id = -1;
                return FalmErr::cpmBufReqErr;
            }

            CpmBuffer &buf = buffer[first_vacant];
            if (hdc == HDC::Host) {
                falmErrCheckMacro(falmMallocDevice(&buf.packer, reqsize));
                falmErrCheckMacro(falmMallocPinned(&buf.ptr, reqsize));
            } else if (hdc == HDC::Device) {
                falmErrCheckMacro(falmMallocDevice(&buf.ptr, reqsize));
            }
            printf("buffer %d -> %lu\n", first_vacant, reqsize);
            buf.hdc = hdc;
            buf.capacity = reqsize;
            buf.map = map;
            buf.count = count;
            buf.color = color;
            buf.active = true;
            buf.width = width;
            *buf_id = first_vacant;
            max_used_idx = first_vacant;
            return FalmErr::success;
        }
    }

    int mark_release(int i) {
        buffer[i].active = false;
        return FalmErr::success;
    }

    int release_all() {
        for (int i = 0; i < NBUFFER; i ++) {
            if(release(i)) {
                return FalmErr::cpmBufReleaseErr;
            }
        }
        return FalmErr::success;
    }
};

}