#ifndef FALM_RMCP_BLADEHANDLER_H
#define FALM_RMCP_BLADEHANDLER_H

#include <fstream>
#include <string>
#include "../nlohmann/json.hpp"
#include "../util.h"

namespace Falm {

struct BHFrame {
    REAL *r;
    REAL *attack;
    REAL *chord;
    REAL *twist;
    REAL *cl;
    REAL *cd;
    INT rcount, acount;
    FLAG  hdc;

    BHFrame() : 
        r(nullptr), 
        attack(nullptr), 
        chord(nullptr), 
        twist(nullptr), 
        cl(nullptr), 
        cd(nullptr), 
        rcount(0),
        acount(0), 
        hdc(HDC::Empty) 
    {}

    __host__ __device__ REAL &getcl(INT rid, INT aid) {
        return cl[rid*acount + aid];
    }

    __host__ __device__ REAL &getcd(INT rid, INT aid) {
        return cd[rid*acount + aid];
    }

    void alloc(INT _rcount, INT _attackcount, FLAG _hdc) {
        rcount = _rcount;
        acount = _attackcount;
        hdc = _hdc;
        if (hdc == HDC::Host) {
            falmErrCheckMacro(falmMalloc((void**)&r, sizeof(REAL)*rcount));
            falmErrCheckMacro(falmMalloc((void**)&attack, sizeof(REAL)*acount));
            falmErrCheckMacro(falmMalloc((void**)&chord, sizeof(REAL)*rcount));
            falmErrCheckMacro(falmMalloc((void**)&twist, sizeof(REAL)*rcount));
            falmErrCheckMacro(falmMalloc((void**)&cl, sizeof(REAL)*rcount*acount));
            falmErrCheckMacro(falmMalloc((void**)&cd, sizeof(REAL)*rcount*acount));
        } else if (hdc == HDC::Device) {
            falmErrCheckMacro(falmMallocDevice((void**)&r, sizeof(REAL)*rcount));
            falmErrCheckMacro(falmMallocDevice((void**)&attack, sizeof(REAL)*acount));
            falmErrCheckMacro(falmMallocDevice((void**)&chord, sizeof(REAL)*rcount));
            falmErrCheckMacro(falmMallocDevice((void**)&twist, sizeof(REAL)*rcount));
            falmErrCheckMacro(falmMallocDevice((void**)&cl, sizeof(REAL)*rcount*acount));
            falmErrCheckMacro(falmMallocDevice((void**)&cd, sizeof(REAL)*rcount*acount));
        }
    }

    void release() {
        if (hdc == HDC::Host) {
            falmErrCheckMacro(falmFree(r));
            falmErrCheckMacro(falmFree(attack));
            falmErrCheckMacro(falmFree(chord));
            falmErrCheckMacro(falmFree(twist));
            falmErrCheckMacro(falmFree(cl));
            falmErrCheckMacro(falmFree(cd));
        } else if (hdc == HDC::Device) {
            falmErrCheckMacro(falmFreeDevice(r));
            falmErrCheckMacro(falmFreeDevice(attack));
            falmErrCheckMacro(falmFreeDevice(chord));
            falmErrCheckMacro(falmFreeDevice(twist));
            falmErrCheckMacro(falmFreeDevice(cl));
            falmErrCheckMacro(falmFreeDevice(cd));
        }
        hdc = HDC::Empty;
    }

    __host__ __device__ INT find_position(REAL *metric, REAL x, INT size) {
        if (x < metric[0]) {
            return -1;
        } else if (x >= metric[size-1]) {
            return size-1;
        } else {
            for (INT i = 0; i < size-1; i ++) {
                if (metric[i] <= x && metric[i+1] > x) {
                    return i;
                }
            }
            return -2;
        }
    }

    __host__ __device__ REAL interpolate(REAL *metric, REAL *data, REAL x, INT i, INT size) {
        if (i < 0) {
            return data[0];
        } else if (i >= size-1) {
            return data[size-1];
        } else {
            REAL p = (x - metric[i])/(metric[i+1] - metric[i]);
            return (1. - p)*data[i] + p*data[i+1];
        }
    }

    __host__ __device__ void get_airfoil_params(REAL _r, REAL _phi, REAL &_chord, REAL &_twist, REAL &_cl, REAL &_cd) {
        INT rid = find_position(r, _r, rcount);
        _chord = interpolate(r, chord, _r, rid, rcount);
        _twist = interpolate(r, twist, _r, rid, rcount);
        REAL _attack = _phi - _twist;
        INT aid = find_position(attack, _attack, acount);
        if (rid < 0) {
            _cl = interpolate(attack, &getcl(0,0), _attack, aid, acount);
            _cd = interpolate(attack, &getcd(0,0), _attack, aid, acount);
        } else if (rid >= rcount-1) {
            _cl = interpolate(attack, &getcl(rcount-1,0), _attack, aid, acount);
            _cd = interpolate(attack, &getcd(rcount-1,0), _attack, aid, acount);
        } else {
            REAL p = (_attack - attack[aid])/(attack[aid+1] - attack[aid]);
            REAL cl0 = interpolate(attack, &getcl(rid  ,0), _attack, aid, acount);
            REAL cl1 = interpolate(attack, &getcl(rid+1,0), _attack, aid, acount);
            REAL cd0 = interpolate(attack, &getcd(rid  ,0), _attack, aid, acount);
            REAL cd1 = interpolate(attack, &getcd(rid+1,0), _attack, aid, acount);
            _cl = (1. - p)*cl0 + p*cl1;
            _cd = (1. - p)*cd0 + p*cd1;
        }
    }
};

struct BladeHandler {
    BHFrame host, dev, *devptr;
    INT rcount, acount;
    FLAG hdc;

    BladeHandler() : host(), dev(), devptr(nullptr), rcount(0), acount(0), hdc(HDC::Empty) {}

    REAL &getcl(INT rid, INT aid) {
        return host.getcl(rid, aid);
    }

    REAL &getcd(INT rid, INT aid) {
        return host.getcd(rid, aid);
    }

    void alloc(
        std::string bpfname
    ) {
        std::ifstream ifs(bpfname);
        json bpjson = json::parse(ifs);
        json aflist = bpjson["airfoils"];
        json atlist = bpjson["attacks"];
        int _rcount = aflist.size();
        int _acount = atlist.size();
        host.alloc(_rcount, _acount, HDC::Host);
        dev.alloc(_rcount, _acount, HDC::Device);
        rcount = _rcount;
        acount = _acount;
        hdc = HDC::HstDev;
        ifs.close();

        for (INT i = 0; i < rcount; i ++) {
            auto af = aflist[i];
            host.r[i] = af["r/R"].get<REAL>();
            host.chord[i] = af["chord/R"].get<REAL>();
            host.twist[i] = af["twist"].get<REAL>();
            auto cllist = af["Cl"];
            auto cdlist = af["Cd"];
            for (INT j = 0; j < acount; j ++) {
                getcl(i, j) = cllist[j].get<REAL>();
                getcd(i, j) = cdlist[j].get<REAL>();
            }
        }
        for (INT j = 0; j < acount; j ++) {
            host.attack[j] = atlist[j].get<REAL>();
        }

        falmErrCheckMacro(falmMemcpy(dev.r, host.r, sizeof(REAL)*rcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.attack, host.attack, sizeof(REAL)*acount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.chord, host.chord, sizeof(REAL)*rcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.twist, host.twist, sizeof(REAL)*rcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.cl, host.cl, sizeof(REAL)*rcount*acount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.cd, host.cd, sizeof(REAL)*rcount*acount, MCP::Hst2Dev));

        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(BHFrame)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(BHFrame), MCP::Hst2Dev));
    }

    void release() {
        host.release();
        dev.release();
        falmErrCheckMacro(falmFreeDevice(devptr));
        hdc = HDC::Empty;
    }
};


}

#endif