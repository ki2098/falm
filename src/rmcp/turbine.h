#ifndef FALM_TURBINE_H
#define FALM_TURBINE_H

#include <math.h>
#include "../devdefine.h"
#include "../util.h"
#include "../flag.h"

namespace Falm {

namespace Rmcp {

struct RmcpTurbine {
    Real  torque = 0;
    Real  cpower = 0;
    Real3    pos = {{0, 0, 0}};
    Real3 rotpos = {{0, 0, 0}};
    Real       R = 1.0;
    Real   width;
    Real   thick;
    Real     tip;
    Real     hub;
    Real    roll = 0.0;
    Real   pitch = 0.0;
    Real     yaw = 0.0;
    VECTOR<Real, 6> chord_a;
    VECTOR<Real, 6> angle_a;

    __host__ __device__ Real chord(Real r) {
        Real r2 = r * r;
        Real r3 = r * r2;
        Real r4 = r * r3;
        Real r5 = r * r4;
        return chord_a[0] + chord_a[1] * r + chord_a[2] * r2 + chord_a[3] * r3 + chord_a[4] * r4 + chord_a[5] * r5;
    }

    __host__ __device__ Real angle(Real r) {
        Real r2 = r * r;
        Real r3 = r * r2;
        Real r4 = r * r3;
        Real r5 = r * r4;
        return angle_a[0] + angle_a[1] * r + angle_a[2] * r2 + angle_a[3] * r3 + angle_a[4] * r4 + angle_a[5] * r5;
    }

    __host__ __device__ Real3 transform(const Real3 &vxyz) {
        Real s1, s2, s3, c1, c2, c3;
        s1 = sin(roll);
        s2 = sin(pitch);
        s3 = sin(yaw);
        c1 = cos(roll);
        c2 = cos(pitch);
        c3 = cos(yaw);
        Real x1 = vxyz[0], y1 = vxyz[1], z1 = vxyz[2];
        Real x2 = (c2 * c3               ) * x1 + (c2 * s3               ) * y1 + (- s2   ) * z1;
        Real y2 = (s1 * s2 * c3 - c1 * s3) * x1 + (s1 * s2 * s3 + c1 * c3) * y1 + (s1 * c2) * z1;
        Real z2 = (c1 * s2 * c3 + s1 * s3) * x1 + (c1 * s2 * s3 - s1 * c3) * y1 + (c1 * c2) * z1;
        return {{x2, y2, z2}};
    }

    __host__ __device__ Real3 invert_transform(const Real3 &vxyz2) {
        Real s1, s2, s3, c1, c2, c3;
        s1 = sin(roll);
        s2 = sin(pitch);
        s3 = sin(yaw);
        c1 = cos(roll);
        c2 = cos(pitch);
        c3 = cos(yaw);
        Real x2 = vxyz2[0], y2 = vxyz2[1], z2 = vxyz2[2];
        Real x1 = (c2 * c3) * x2 + (s1 * s2 * c3 - c1 * s3) * y2 + (c1 * s2 * c3 + s1 * s3) * z2;
        Real y1 = (c2 * s3) * x2 + (s1 * s2 * s3 + c1 * c3) * y2 + (c1 * s2 * s3 - s1 * c3) * z2;
        Real z1 = (- s2   ) * x2 + (s1 * c2               ) * y2 + (c1 * c2               ) * z2;
        return {{x1, y1, z1}};
    }

    __host__ __device__ Real Cd(Real alpha) {
        return 0.02;
    }
    
    __host__ __device__ Real Cl(Real alpha) {
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
    Flag     hdctype;
    Int     nTurbine;

    RmcpTurbineArray(const RmcpTurbineArray &_wf) = delete;
    RmcpTurbineArray &operator=(const RmcpTurbineArray &_wf) = delete;

    RmcpTurbineArray(Int _n) {
        nTurbine = _n;
        hdctype  = HDC::Host;
        falmErrCheckMacro(falmMalloc((void**)&tptr, sizeof(RmcpTurbine) * nTurbine));
        tdevptr  = nullptr;
    }

    void sync(Flag _mcptype) {
        if (_mcptype == MCP::Hst2Dev) {
            if (hdctype & HDC::Device) {
                falmErrCheckMacro(falmMemcpy(tdevptr, tptr, sizeof(RmcpTurbine) * nTurbine, MCP::Hst2Dev));
            } else {
                falmErrCheckMacro(falmMallocDevice((void**)&tdevptr, sizeof(RmcpTurbine) * nTurbine));
                falmErrCheckMacro(falmMemcpy(tdevptr, tptr, sizeof(RmcpTurbine) * nTurbine, MCP::Hst2Dev));
                hdctype &= HDC::Device;
            }
        } else if (_mcptype == MCP::Dev2Hst) {
            if (hdctype & HDC::Host) {
                falmErrCheckMacro(falmMemcpy(tptr, tdevptr, sizeof(RmcpTurbine) * nTurbine, MCP::Dev2Hst));
            } else {
                falmErrCheckMacro(falmMalloc((void**)&tptr, sizeof(RmcpTurbine) * nTurbine));
                falmErrCheckMacro(falmMemcpy(tptr, tdevptr, sizeof(RmcpTurbine) * nTurbine, MCP::Dev2Hst));
                hdctype &= HDC::Host;
            }
        }
    }

    void release(Flag _hdctype) {
        if (_hdctype & HDC::Host) {
            falmErrCheckMacro(falmFree(tptr));
            hdctype &= ~(HDC::Host);
            tptr = nullptr;
        }
        if (_hdctype & HDC::Device) {
            falmErrCheckMacro(falmFreeDevice(tdevptr));
            hdctype &= ~(HDC::Device);
            tdevptr = nullptr;
        }
    }

    ~RmcpTurbineArray() {
        if (hdctype & HDC::Device) {
            falmErrCheckMacro(falmFreeDevice(tdevptr));
        }
        if (hdctype & HDC::Host) {
            falmErrCheckMacro(falmFree(tptr));
        }
    }

    RmcpTurbine &operator[](Int i) {
        return tptr[i];
    }
};

}

}

#endif
