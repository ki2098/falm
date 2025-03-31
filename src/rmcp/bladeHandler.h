#ifndef FALM_RMCP_BLADEHANDLER_H
#define FALM_RMCP_BLADEHANDLER_H

#include <fstream>
#include <string>
#include "../nlohmann/json.hpp"
#include "../util.h"

namespace Falm {

namespace Rmcp {

struct BHFrame {
    Real *r;
    Real *attack;
    Real *chord;
    Real *twist;
    Real *cl;
    Real *cd;
    Int rcount, acount;
    Flag  hdc;

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

    __host__ __device__ Real &getcl(Int rid, Int aid) {
        return cl[rid*acount + aid];
    }

    __host__ __device__ Real &getcd(Int rid, Int aid) {
        return cd[rid*acount + aid];
    }

    void alloc(Int _rcount, Int _attackcount, Flag _hdc) {
        rcount = _rcount;
        acount = _attackcount;
        hdc = _hdc;
        if (hdc == HDC::Host) {
            falmErrCheckMacro(falmMalloc((void**)&r, sizeof(Real)*rcount));
            falmErrCheckMacro(falmMalloc((void**)&attack, sizeof(Real)*acount));
            falmErrCheckMacro(falmMalloc((void**)&chord, sizeof(Real)*rcount));
            falmErrCheckMacro(falmMalloc((void**)&twist, sizeof(Real)*rcount));
            falmErrCheckMacro(falmMalloc((void**)&cl, sizeof(Real)*rcount*acount));
            falmErrCheckMacro(falmMalloc((void**)&cd, sizeof(Real)*rcount*acount));
        } else if (hdc == HDC::Device) {
            falmErrCheckMacro(falmMallocDevice((void**)&r, sizeof(Real)*rcount));
            falmErrCheckMacro(falmMallocDevice((void**)&attack, sizeof(Real)*acount));
            falmErrCheckMacro(falmMallocDevice((void**)&chord, sizeof(Real)*rcount));
            falmErrCheckMacro(falmMallocDevice((void**)&twist, sizeof(Real)*rcount));
            falmErrCheckMacro(falmMallocDevice((void**)&cl, sizeof(Real)*rcount*acount));
            falmErrCheckMacro(falmMallocDevice((void**)&cd, sizeof(Real)*rcount*acount));
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

    __host__ __device__ Int find_position(Real *metric, Real x, Int size) {
        if (x < metric[0]) {
            return -1;
        } else if (x >= metric[size-1]) {
            return size-1;
        } else {
            for (Int i = 0; i < size-1; i ++) {
                if (metric[i] <= x && metric[i+1] > x) {
                    return i;
                }
            }
            return -2;
        }
    }

    __host__ __device__ Real interpolate(Real *metric, Real *data, Real x, Int i, Int size) {
        if (i < 0) {
            return data[0];
        } else if (i >= size-1) {
            return data[size-1];
        } else {
            Real p = (x - metric[i])/(metric[i+1] - metric[i]);
            return (1. - p)*data[i] + p*data[i+1];
        }
    }

    __host__ __device__ void get_airfoil_params(Real _r, Real _phi, Real &_chord, Real &_twist, Real &_cl, Real &_cd) {
        Int rid = find_position(r, _r, rcount);
        _chord = interpolate(r, chord, _r, rid, rcount);
        _twist = interpolate(r, twist, _r, rid, rcount);
        Real _attack = _phi - _twist;
        Int aid = find_position(attack, _attack, acount);
        if (rid < 0) {
            _cl = interpolate(attack, &getcl(0,0), _attack, aid, acount);
            _cd = interpolate(attack, &getcd(0,0), _attack, aid, acount);
        } else if (rid >= rcount-1) {
            _cl = interpolate(attack, &getcl(rcount-1,0), _attack, aid, acount);
            _cd = interpolate(attack, &getcd(rcount-1,0), _attack, aid, acount);
        } else {
            Real p = (_r - r[rid])/(r[rid+1] - r[rid]);
            Real cl0 = interpolate(attack, &getcl(rid  ,0), _attack, aid, acount);
            Real cl1 = interpolate(attack, &getcl(rid+1,0), _attack, aid, acount);
            Real cd0 = interpolate(attack, &getcd(rid  ,0), _attack, aid, acount);
            Real cd1 = interpolate(attack, &getcd(rid+1,0), _attack, aid, acount);
            _cl = (1. - p)*cl0 + p*cl1;
            _cd = (1. - p)*cd0 + p*cd1;
        }
    }
};

struct BladeHandler {
    BHFrame host, dev, *devptr;
    Int rcount, acount;
    Flag hdc;
    std::string property_file_path;

    BladeHandler() : host(), dev(), devptr(nullptr), rcount(0), acount(0), hdc(HDC::Empty) {}

    Real &getcl(Int rid, Int aid) {
        return host.getcl(rid, aid);
    }

    Real &getcd(Int rid, Int aid) {
        return host.getcd(rid, aid);
    }

    void alloc(
        std::string bpfname
    ) {
        // printf("BLADE PROPERTY INFO\n");
        // printf("\tPath %s\n", bpfname.c_str());
        property_file_path = bpfname;
        std::ifstream ifs(bpfname);
        Json bpjson = Json::parse(ifs);
        Json aflist = bpjson["airfoils"];
        Json atlist = bpjson["attacks"];
        int _rcount = aflist.size();
        int _acount = atlist.size();
        host.alloc(_rcount, _acount, HDC::Host);
        dev.alloc(_rcount, _acount, HDC::Device);
        rcount = _rcount;
        acount = _acount;
        hdc = HDC::HstDev;
        ifs.close();

        for (Int i = 0; i < rcount; i ++) {
            auto af = aflist[i];
            host.r[i] = af["r/R"].get<Real>();
            host.chord[i] = af["chord/R"].get<Real>();
            host.twist[i] = af["twist[deg]"].get<Real>();
            auto cllist = af["Cl"];
            auto cdlist = af["Cd"];
            for (Int j = 0; j < acount; j ++) {
                getcl(i, j) = cllist[j].get<Real>();
                getcd(i, j) = cdlist[j].get<Real>();
            }
        }
        for (Int j = 0; j < acount; j ++) {
            host.attack[j] = atlist[j].get<Real>();
        }

        falmErrCheckMacro(falmMemcpy(dev.r, host.r, sizeof(Real)*rcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.attack, host.attack, sizeof(Real)*acount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.chord, host.chord, sizeof(Real)*rcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.twist, host.twist, sizeof(Real)*rcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.cl, host.cl, sizeof(Real)*rcount*acount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.cd, host.cd, sizeof(Real)*rcount*acount, MCP::Hst2Dev));

        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(BHFrame)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(BHFrame), MCP::Hst2Dev));
    }

    void release() {
        host.release();
        dev.release();
        falmErrCheckMacro(falmFreeDevice(devptr));
        hdc = HDC::Empty;
    }

    void get_airfoil_params(Real _r, Real _phi, Real &_chord, Real &_twist, Real &_cl, Real &_cd) {
        host.get_airfoil_params(_r, _phi, _chord, _twist, _cl, _cd);
    }
};

}

}

#endif