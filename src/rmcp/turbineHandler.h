#ifndef FALM_RMCP_TURBINEHANDLER_H
#define FALM_RMCP_TURBINEHANDLER_H

#include <string>
#include "../nlohmann/json.hpp"
#include "../util.h"
#include "../falmath.h"

namespace Falm {

namespace Rmcp {

struct TurbineFrame {
    REAL3      *base;
    REAL3      *base_velocity;
    REAL3      *hub;
    REAL3      *angle;
    REAL3      *angular_velocity;
    EulerAngle *angle_type;
    REAL       *tip_rate;
    REAL       *torque;
    REAL        radius;
    REAL        hub_radius;
    size_t      n_turbine, n_blade;
    FLAG        hdc;

    TurbineFrame() :
        base(nullptr),
        base_velocity(nullptr),
        hub(nullptr),
        angle(nullptr),
        angular_velocity(nullptr),
        angle_type(nullptr),
        tip_rate(nullptr),
        torque(nullptr),
        hdc(HDC::Empty)
    {}

    __host__ __device__ size_t id(size_t tid, size_t nid) {
        return nid*n_turbine + tid;
    }

    void alloc(size_t _tcount, size_t _bcount, REAL _r, REAL _hr, FLAG _hdc) {
        n_turbine = _tcount;
        n_blade = _bcount;
        radius = _r;
        hub_radius = _hr;
        hdc = _hdc;
        if (hdc == HDC::Host) {
            falmErrCheckMacro(falmMalloc((void**)&base, sizeof(REAL3)*n_turbine));
            falmErrCheckMacro(falmMalloc((void**)&base_velocity, sizeof(REAL3)*n_turbine));
            falmErrCheckMacro(falmMalloc((void**)&hub, sizeof(REAL3)*n_turbine));
            falmErrCheckMacro(falmMalloc((void**)&angle, sizeof(REAL3)*n_turbine));
            falmErrCheckMacro(falmMalloc((void**)&angular_velocity, sizeof(REAL3)*n_turbine));
            falmErrCheckMacro(falmMalloc((void**)&angle_type, sizeof(EulerAngle)*n_turbine));
            falmErrCheckMacro(falmMalloc((void**)&tip_rate, sizeof(REAL)*n_turbine));
            falmErrCheckMacro(falmMalloc((void**)&torque, sizeof(REAL)*n_turbine));
        } else if (hdc == HDC::Device) {
            falmErrCheckMacro(falmMallocDevice((void**)&base, sizeof(REAL3)*n_turbine));
            falmErrCheckMacro(falmMallocDevice((void**)&base_velocity, sizeof(REAL3)*n_turbine));
            falmErrCheckMacro(falmMallocDevice((void**)&hub, sizeof(REAL3)*n_turbine));
            falmErrCheckMacro(falmMallocDevice((void**)&angle, sizeof(REAL3)*n_turbine));
            falmErrCheckMacro(falmMallocDevice((void**)&angular_velocity, sizeof(REAL3)*n_turbine));
            falmErrCheckMacro(falmMallocDevice((void**)&angle_type, sizeof(EulerAngle)*n_turbine));
            falmErrCheckMacro(falmMallocDevice((void**)&tip_rate, sizeof(REAL)*n_turbine));
            falmErrCheckMacro(falmMallocDevice((void**)&torque, sizeof(REAL)*n_turbine));
        }
    }

    void release() {
        if (hdc == HDC::Host) {
            falmErrCheckMacro(falmFree(base));
            falmErrCheckMacro(falmFree(base_velocity));
            falmErrCheckMacro(falmFree(hub));
            falmErrCheckMacro(falmFree(angle));
            falmErrCheckMacro(falmFree(angular_velocity));
            falmErrCheckMacro(falmFree(angle_type));
            falmErrCheckMacro(falmFree(tip_rate));
            falmErrCheckMacro(falmFree(torque));
        } else if (hdc == HDC::Device) {
            falmErrCheckMacro(falmFreeDevice(base));
            falmErrCheckMacro(falmFreeDevice(base_velocity));
            falmErrCheckMacro(falmFreeDevice(hub));
            falmErrCheckMacro(falmFreeDevice(angle));
            falmErrCheckMacro(falmFreeDevice(angular_velocity));
            falmErrCheckMacro(falmFreeDevice(angle_type));
            falmErrCheckMacro(falmFreeDevice(tip_rate));
            falmErrCheckMacro(falmFreeDevice(torque));
        }
        hdc = HDC::Empty;
    }
};

struct TurbineHandler {
    TurbineFrame host, dev, *devptr;
    size_t       n_turbine, n_blade;
    REAL         radius;
    REAL         hub_radius;
    FLAG         hdc;
    json         turbine_param;

    TurbineHandler() :
        host(),
        dev(),
        devptr(nullptr),
        hdc(HDC::Empty)
    {}

    void alloc(const json &_turbine_param) {
        turbine_param = _turbine_param;
        radius = turbine_param["radius"].get<REAL>();
        n_blade = turbine_param["bladeNumber"].get<size_t>();
        auto turbine_array = turbine_param["turbineArray"];
        if (turbine_param.contains("hubRadius")) {
            hub_radius = turbine_param["hubRadius"].get<REAL>();
        } else {
            hub_radius = 0;
        }
        n_turbine = turbine_array.size();
        host.alloc(n_turbine, n_blade, radius, hub_radius, HDC::Host);
        dev.alloc(n_turbine, n_blade, radius, hub_radius, HDC::Device);
        for (int tid = 0; tid < n_turbine; tid ++) {
            auto turbine_json = turbine_array[tid];
            host.hub[tid][0] = - turbine_param["overhang"].get<REAL>();
            host.hub[tid][1] =   0;
            host.hub[tid][2] =   turbine_param["tower"].get<REAL>();
            host.tip_rate[tid] = turbine_json["tipRate"].get<REAL>();
            
            auto tmp = turbine_json["baseLocation"];
            for (int i = 0; i < 3; i ++) {
                if (tmp[i].is_number()) {
                    host.base[tid][i] = tmp[i].get<REAL>();
                    host.base_velocity[tid][i] = 0;
                }
            }

            tmp = turbine_json["angle"];
            int angle_switch[] = {0, 0, 0};
            for (int i = 0; i < 3; i ++) {
                if (tmp[i].is_number()) {
                    host.angle[tid][i] = deg2rad(tmp[i].get<REAL>());
                    host.angular_velocity[tid][i] = 0;
                    if (tmp[i].get<REAL>() != 0) {
                        angle_switch[i] = 1;
                    }
                } else {
                    angle_switch[i] = 1;
                }
            }
            if (angle_switch[0] + angle_switch[1] + angle_switch[2] > 1) {
                fprintf(stderr, "Error in turbine %d, only 1 euler angle allowed.\n", tid);
                host.angle_type[tid] = EulerAngle::Empty;
            } else if (angle_switch[0]) {
                host.angle_type[tid] = EulerAngle::Roll;
            } else if (angle_switch[1]) {
                host.angle_type[tid] = EulerAngle::Pitch;
            } else if (angle_switch[2]) {
                host.angle_type[tid] = EulerAngle::Yaw;
            } else {
                host.angle_type[tid] = EulerAngle::Empty;
            }
        }

        falmErrCheckMacro(falmMemcpy(dev.base, host.base, sizeof(REAL3)*n_turbine, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.base_velocity, host.base_velocity, sizeof(REAL3)*n_turbine, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.hub, host.hub, sizeof(REAL3)*n_turbine, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.angle, host.angle, sizeof(REAL3)*n_turbine, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.angular_velocity, host.angular_velocity, sizeof(REAL3)*n_turbine, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.angle_type, host.angle_type, sizeof(EulerAngle)*n_turbine, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.tip_rate, host.tip_rate, sizeof(REAL)*n_turbine, MCP::Hst2Dev));

        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(TurbineFrame)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(TurbineFrame), MCP::Hst2Dev));
        hdc = HDC::HstDev;
    }

    void release() {
        host.release();
        dev.release();
        falmErrCheckMacro(falmFreeDevice(devptr));
        hdc = HDC::Empty;
    }
};

}

}

#endif