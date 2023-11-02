#ifndef FALM_TURBINE_H
#define FALM_TURBINE_H

#include "devdefine.h"
#include "util.h"
#include "flag.h"

namespace Falm {

struct RmcpTurbine {
    REAL  torque;
    REAL  cpower;
    REAL3    pos;
    REAL       R;
    REAL       W;
    REAL       D;
    REAL     tip;
    REAL     hub;
    REAL chord_a[6];
    REAL angle_a[6];

    __host__ __device__ REAL chord(REAL r) {
        REAL r2 = r * r;
        REAL r3 = r * r2;
        REAL r4 = r * r3;
        REAL r5 = r * r4;
        return chord_a[0] + chord_a[1] * r + chord_a[2] * r2 + chord_a[3] * r3 + chord_a[4] * r4 + chord_a[5] * r5;
    }

    __host__ __device__ REAL angle(REAL r) {
        REAL r2 = r * r;
        REAL r3 = r * r2;
        REAL r4 = r * r3;
        REAL r5 = r * r4;
        return angle_a[0] + angle_a[1] * r + angle_a[2] * r2 + angle_a[3] * r3 + angle_a[4] * r4 + angle_a[5] * r5;
    }
};

struct RmcpWindfarm {
    RmcpTurbine *tptr;
    RmcpTurbine *tdevptr;
    FLAG     hdctype;
    INT     nTurbine;

    RmcpWindfarm(const RmcpWindfarm &_wf) = delete;
    RmcpWindfarm &operator=(const RmcpWindfarm &_wf) = delete;

    RmcpWindfarm(INT _n) {
        nTurbine = _n;
        hdctype  = HDCType::Host;
        tptr     = (RmcpTurbine*)falmMallocPinned(sizeof(RmcpTurbine) * nTurbine);
        tdevptr  = nullptr;
    }

    void sync(FLAG _mcptype) {
        if (_mcptype == MCpType::Hst2Dev) {
            if (hdctype & HDCType::Device) {
                falmMemcpy(tdevptr, tptr, sizeof(RmcpTurbine) * nTurbine, MCpType::Hst2Dev);
            } else {
                tdevptr = (RmcpTurbine *)falmMallocDevice(sizeof(RmcpTurbine) * nTurbine);
                falmMemcpy(tdevptr, tptr, sizeof(RmcpTurbine) * nTurbine, MCpType::Hst2Dev);
            }
        } else if (_mcptype == MCpType::Dev2Hst) {
            if (hdctype & HDCType::Host) {
                falmMemcpy(tptr, tdevptr, sizeof(RmcpTurbine) * nTurbine, MCpType::Dev2Hst);
            } else {
                tptr = (RmcpTurbine*)falmMallocPinned(sizeof(RmcpTurbine) * nTurbine);
                falmMemcpy(tptr, tdevptr, sizeof(RmcpTurbine) * nTurbine, MCpType::Dev2Hst);
            }
        }
    }

    void release(FLAG _hdctype) {
        if (_hdctype & HDCType::Host) {
            falmFreePinned(tptr);
            hdctype &= ~(HDCType::Host);
            tptr = nullptr;
        }
        if (_hdctype & HDCType::Device) {
            falmFreeDevice(tdevptr);
            hdctype &= ~(HDCType::Device);
            tdevptr = nullptr;
        }
    }

    ~RmcpWindfarm() {
        if (hdctype & HDCType::Device) {
            falmFreeDevice(tdevptr);
        }
        if (hdctype & HDCType::Host) {
            falmFreePinned(tptr);
        }
    }

    RmcpTurbine &operator[](INT i) {
        return tptr[i];
    }
};

}

#endif