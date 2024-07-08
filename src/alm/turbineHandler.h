#ifndef FALM_ALM_TURBINEHANDLER_H
#define FALM_ALM_TURBINEHANDLER_H

#include "../util.h"

namespace Falm {

struct TurbineFrame {
    size_t turbinecount, bladecount, bladepointcount;
    FLAG hdc;
    REAL radius;
    REAL *base; // turbinecount*3
    REAL *basevelocity; // turbinecount*3
    REAL *hub; // turbinecount*3
    REAL *angle; // turbinecount
    REAL *angularVelocity; // turbinecount*3
    REAL *tiprate; // turbinecount

    TurbineFrame() : 
        base(nullptr),
        basevelocity(nullptr),
        hub(nullptr),
        angle(nullptr),
        angularVelocity(nullptr),
        tiprate(nullptr),
        hdc(HDC::Empty)
    {}

    __host__ __device__ size_t id(size_t turbineid, size_t n) {
        return turbineid + n*turbinecount;
    }

    void alloc(size_t _turbinecount, size_t _bladecount, size_t _bladepointcount, REAL _radius, FLAG _hdc) {
        turbinecount = _turbinecount;
        bladecount = _bladecount;
        bladepointcount = _bladepointcount;
        radius = _radius;
        hdc = _hdc;
        if (hdc == HDC::Host) {
            falmErrCheckMacro(falmMalloc((void**)&base, sizeof(REAL)*turbinecount*3));
            falmErrCheckMacro(falmMalloc((void**)&basevelocity, sizeof(REAL)*turbinecount*3));
            falmErrCheckMacro(falmMalloc((void**)&hub, sizeof(REAL)*turbinecount*3));
            falmErrCheckMacro(falmMalloc((void**)&angle, sizeof(REAL)*turbinecount*3));
            falmErrCheckMacro(falmMalloc((void**)&angularVelocity, sizeof(REAL)*turbinecount*3));
            falmErrCheckMacro(falmMalloc((void**)&tiprate, sizeof(REAL)*turbinecount));
        } else if (hdc == HDC::Device) {
            falmErrCheckMacro(falmMallocDevice((void**)&base, sizeof(REAL)*turbinecount*3));
            falmErrCheckMacro(falmMallocDevice((void**)&basevelocity, sizeof(REAL)*turbinecount*3));
            falmErrCheckMacro(falmMallocDevice((void**)&hub, sizeof(REAL)*turbinecount*3));
            falmErrCheckMacro(falmMallocDevice((void**)&angle, sizeof(REAL)*turbinecount*3));
            falmErrCheckMacro(falmMallocDevice((void**)&angularVelocity, sizeof(REAL)*turbinecount*3));
            falmErrCheckMacro(falmMallocDevice((void**)&tiprate, sizeof(REAL)*turbinecount));
        }
    }

    void release() {
        if (hdc == HDC::Host) {
            falmErrCheckMacro(falmFree(base));
            falmErrCheckMacro(falmFree(basevelocity));
            falmErrCheckMacro(falmFree(hub));
            falmErrCheckMacro(falmFree(angle));
            falmErrCheckMacro(falmFree(angularVelocity));
            falmErrCheckMacro(falmFree(tiprate));
        } else if (hdc == HDC::Device) {
            falmErrCheckMacro(falmFreeDevice(base));
            falmErrCheckMacro(falmFreeDevice(basevelocity));
            falmErrCheckMacro(falmFreeDevice(hub));
            falmErrCheckMacro(falmFreeDevice(angle));
            falmErrCheckMacro(falmFreeDevice(angularVelocity));
            falmErrCheckMacro(falmFreeDevice(tiprate));
        }
        hdc = HDC::Empty;
    }
};

struct TurbineHandler {
    TurbineFrame host, dev, *devptr;
    size_t turbinecount, bladecount, bladepointcount;
    FLAG hdc;
    REAL radius;

    size_t id(size_t turbineid, size_t n) {
        return turbineid + n*turbinecount;
    }

    void alloc(size_t _turbinecount, size_t _bladecount, size_t _bladepointcount, REAL _radius) {
        turbinecount = _turbinecount;
        bladecount = _bladecount;
        bladepointcount = _bladepointcount;
        radius = _radius;
        hdc = HDC::HstDev;
    }
};

}

#endif