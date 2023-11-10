#ifndef FALM_TURBINE_H
#define FALM_TURBINE_H

#include <math.h>
#include "../devdefine.h"
#include "../util.h"
#include "../flag.h"

namespace Falm {

struct RmcpTurbine {
    REAL  torque = 0;
    REAL  cpower = 0;
    REAL3    pos = {0, 0, 0};
    REAL3 rotpos = {0, 0, 0};
    REAL       R = 1.0;
    REAL   width;
    REAL   thick;
    REAL     tip;
    REAL     hub;
    REAL    roll = 0.0;
    REAL   pitch = 0.0;
    REAL     yaw = 0.0;
    VECTOR<REAL, 6> chord_a;
    VECTOR<REAL, 6> angle_a;

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

    __host__ __device__ REAL3 transform(const REAL3 &vxyz) {
        REAL s1, s2, s3, c1, c2, c3;
        s1 = sin(roll);
        s2 = sin(pitch);
        s3 = sin(yaw);
        c1 = cos(roll);
        c2 = cos(pitch);
        c3 = cos(yaw);
        REAL x1 = vxyz[0], y1 = vxyz[1], z1 = vxyz[2];
        REAL x2 = (c2 * c3               ) * x1 + (c2 * s3               ) * y1 + (- s2   ) * z1;
        REAL y2 = (s1 * s2 * c3 - c1 * s3) * x1 + (s1 * s2 * s3 + c1 * c3) * y1 + (s1 * c2) * z1;
        REAL z2 = (c1 * s2 * c3 + s1 * s3) * x1 + (c1 * s2 * s3 - s1 * c3) * y1 + (c1 * c2) * z1;
        return {x2, y2, z2};
    }

    __host__ __device__ REAL3 invert_transform(const REAL3 &vxyz2) {
        REAL s1, s2, s3, c1, c2, c3;
        s1 = sin(roll);
        s2 = sin(pitch);
        s3 = sin(yaw);
        c1 = cos(roll);
        c2 = cos(pitch);
        c3 = cos(yaw);
        REAL x2 = vxyz2[0], y2 = vxyz2[1], z2 = vxyz2[2];
        REAL x1 = (c2 * c3) * x2 + (s1 * s2 * c3 - c1 * s3) * y2 + (c1 * s2 * c3 + s1 * s3) * z2;
        REAL y1 = (c2 * s3) * x2 + (s1 * s2 * s3 + c1 * c3) * y2 + (c1 * s2 * s3 - s1 * c3) * z2;
        REAL z1 = (- s2   ) * x2 + (s1 * c2               ) * y2 + (c1 * c2               ) * z2;
        return {x1, y1, z1};
    }

    __host__ __device__ REAL Cd(REAL alpha) {
        return 0.02;
    }
    
    __host__ __device__ REAL Cl(REAL alpha) {
        if (alpha > - 6 && alpha < 7) {
            return 0.39087 + 0.10278 * alpha;
        } else if (alpha > 7) {
            return 1.0;
        } else {
            return - 0.02;
        }
    }
};

struct RmcpTurbineArray {
    RmcpTurbine *tptr;
    RmcpTurbine *tdevptr;
    FLAG     hdctype;
    INT     nTurbine;

    RmcpTurbineArray(const RmcpTurbineArray &_wf) = delete;
    RmcpTurbineArray &operator=(const RmcpTurbineArray &_wf) = delete;

    RmcpTurbineArray(INT _n) {
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
                hdctype &= HDCType::Device;
            }
        } else if (_mcptype == MCpType::Dev2Hst) {
            if (hdctype & HDCType::Host) {
                falmMemcpy(tptr, tdevptr, sizeof(RmcpTurbine) * nTurbine, MCpType::Dev2Hst);
            } else {
                tptr = (RmcpTurbine*)falmMallocPinned(sizeof(RmcpTurbine) * nTurbine);
                falmMemcpy(tptr, tdevptr, sizeof(RmcpTurbine) * nTurbine, MCpType::Dev2Hst);
                hdctype &= HDCType::Host;
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

    ~RmcpTurbineArray() {
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