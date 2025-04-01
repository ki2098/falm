#ifndef FALM_ALM_APHANDLER_H
#define FALM_ALM_APHANDLER_H

#include <fstream>
#include <string>
#include "../util.h"

namespace Falm {

namespace Alm {

struct APFrame {
    Real3 *xyz; 
    Int3  *ijk; 
    int  *rank; 
    Real *r; 
    Real *attack; 
    Real *chord; 
    Real *twist; 
    Real *cl; 
    Real *cd; 
    Real3 *force;
    Real *torque;
    Real *thrust;

    size_t apcount;
    size_t attackcount;
    Flag hdc;
    // REAL dr;

    APFrame() :
        xyz(nullptr),
        ijk(nullptr),
        rank(nullptr),
        // turbineid(nullptr),
        // bladeid(nullptr),
        r(nullptr),
        attack(nullptr),
        chord(nullptr),
        twist(nullptr),
        cl(nullptr),
        cd(nullptr),
        force(nullptr),
        torque(nullptr),
        thrust(nullptr),
        apcount(0),
        attackcount(0),
        hdc(HDC::Empty)/* ,
        dr(0) */
    {}

    __host__ __device__ size_t id(size_t apid, size_t atid) const {
        return apid + atid*apcount;
    }

    void alloc(size_t _apcount, size_t _attackcount, Flag _hdc) {
        apcount = _apcount;
        attackcount = _attackcount;

        hdc = _hdc;
        if (hdc == HDC::Host) {
            falmErrCheckMacro(falmMalloc((void**)&xyz      , sizeof(Real3)*apcount));
            falmErrCheckMacro(falmMalloc((void**)&ijk      , sizeof(Int3) *apcount));
            falmErrCheckMacro(falmMalloc((void**)&rank     , sizeof(int) *apcount));
            falmErrCheckMacro(falmMalloc((void**)&r        , sizeof(Real)*apcount));
            falmErrCheckMacro(falmMalloc((void**)&attack   , sizeof(Real)*attackcount));
            falmErrCheckMacro(falmMalloc((void**)&chord    , sizeof(Real)*apcount));
            falmErrCheckMacro(falmMalloc((void**)&twist    , sizeof(Real)*apcount));
            falmErrCheckMacro(falmMalloc((void**)&cl       , sizeof(Real)*apcount*attackcount));
            falmErrCheckMacro(falmMalloc((void**)&cd       , sizeof(Real)*apcount*attackcount));
            falmErrCheckMacro(falmMalloc((void**)&force    , sizeof(Real3)*apcount));
            falmErrCheckMacro(falmMalloc((void**)&torque   , sizeof(Real)*apcount));
            falmErrCheckMacro(falmMalloc((void**)&thrust   , sizeof(Real)*apcount));
        } else if (hdc == HDC::Device) {
            falmErrCheckMacro(falmMallocDevice((void**)&xyz      , sizeof(Real3)*apcount));
            falmErrCheckMacro(falmMallocDevice((void**)&ijk      , sizeof(Int3) *apcount));
            falmErrCheckMacro(falmMallocDevice((void**)&rank     , sizeof(int) *apcount));
            falmErrCheckMacro(falmMallocDevice((void**)&r        , sizeof(Real)*apcount));
            falmErrCheckMacro(falmMallocDevice((void**)&attack   , sizeof(Real)*attackcount));
            falmErrCheckMacro(falmMallocDevice((void**)&chord    , sizeof(Real)*apcount));
            falmErrCheckMacro(falmMallocDevice((void**)&twist    , sizeof(Real)*apcount));
            falmErrCheckMacro(falmMallocDevice((void**)&cl       , sizeof(Real)*apcount*attackcount));
            falmErrCheckMacro(falmMallocDevice((void**)&cd       , sizeof(Real)*apcount*attackcount));
            falmErrCheckMacro(falmMallocDevice((void**)&force    , sizeof(Real3)*apcount));
            falmErrCheckMacro(falmMallocDevice((void**)&torque   , sizeof(Real)*apcount));
            falmErrCheckMacro(falmMallocDevice((void**)&thrust   , sizeof(Real)*apcount));
        }
    }

    void release() {
        if (hdc == HDC::Host) {
            falmErrCheckMacro(falmFree(xyz));
            falmErrCheckMacro(falmFree(ijk));
            falmErrCheckMacro(falmFree(rank));
            falmErrCheckMacro(falmFree(r));
            falmErrCheckMacro(falmFree(attack));
            falmErrCheckMacro(falmFree(chord));
            falmErrCheckMacro(falmFree(twist));
            falmErrCheckMacro(falmFree(cl));
            falmErrCheckMacro(falmFree(cd));
            falmErrCheckMacro(falmFree(force));
            falmErrCheckMacro(falmFree(torque));
            falmErrCheckMacro(falmFree(thrust));
        } else if (hdc == HDC::Device) {
            falmErrCheckMacro(falmFreeDevice(xyz));
            falmErrCheckMacro(falmFreeDevice(ijk));
            falmErrCheckMacro(falmFreeDevice(rank));
            falmErrCheckMacro(falmFreeDevice(r));
            falmErrCheckMacro(falmFreeDevice(attack));
            falmErrCheckMacro(falmFreeDevice(chord));
            falmErrCheckMacro(falmFreeDevice(twist));
            falmErrCheckMacro(falmFreeDevice(cl));
            falmErrCheckMacro(falmFreeDevice(cd));
            falmErrCheckMacro(falmFreeDevice(force));
            falmErrCheckMacro(falmFreeDevice(torque));
            falmErrCheckMacro(falmFreeDevice(thrust));
        }
    }

    // apphi and aptwist are in deg
    __host__ __device__ void get_airfoil_params(size_t apid, Real apphi, Real &apchord, Real &aptwist, Real &apcl, Real &apcd) {
        apchord = chord[apid];
        aptwist = twist[apid];
        Real apattack = apphi - aptwist;
        if (apattack < attack[0]) {
            apcl = cl[id(apid, 0)];
            apcd = cd[id(apid, 0)];
        } else if (apattack >= attack[attackcount-1]) {
            apcl = cl[id(apid, attackcount-1)];
            apcd = cd[id(apid, attackcount-1)];
        } else {
            for (size_t akid = 0; akid < attackcount-1; akid ++) {
                if (attack[akid] <= apattack && attack[akid+1] > apattack) {
                    Real p = (apattack - attack[akid])/(attack[akid+1] - attack[akid]);
                    apcl = (1. - p)*cl[id(apid,akid)] + p*cl[id(apid,akid+1)];
                    apcd = (1. - p)*cd[id(apid,akid)] + p*cd[id(apid,akid+1)];
                    return;
                }
            }
        }
    }

};

struct APHandler {
    APFrame host, dev, *devptr;
    size_t apcount, attackcount;
    Flag hdc;
    // REAL dr;

    APHandler() :
        host(),
        dev(),
        devptr(nullptr),
        apcount(0),
        attackcount(0),
        hdc(HDC::Empty)
    {}

    size_t id(size_t apid, size_t atid) const {
        return apid + atid*apcount;
    }

    void alloc(std::string apfilename) {
        std::ifstream apfile(apfilename);
        auto tmp = Json::parse(apfile);
        apfile.close();
        auto aparrayjson = tmp["aps"];
        auto attackjson = tmp["attacks"];
        apcount = aparrayjson.size();
        attackcount = attackjson.size();
        host.alloc(apcount, attackcount, HDC::Host);
        dev.alloc(apcount, attackcount, HDC::Device);
        hdc = HDC::HstDev;
        for (size_t apid = 0; apid < apcount; apid ++) {
            auto apjson = aparrayjson[apid];
            size_t __id = apjson["id"].get<size_t>();
            if (__id != apid) {
                printf("AP ID ERROR: %lu != %lu\n", apid, __id);
            }
            host.r[apid] = apjson["r"].get<Real>();
            host.chord[apid] = apjson["chord"].get<Real>();
            host.twist[apid] = apjson["twist[deg]"].get<Real>();
            auto cljson = apjson["Cl"];
            auto cdjson = apjson["Cd"];
            for (size_t akid = 0; akid < attackcount; akid ++) {
                host.cl[id(apid, akid)] = cljson[akid].get<Real>();
                host.cd[id(apid, akid)] = cdjson[akid].get<Real>();
            }
        }
        for (size_t akid = 0; akid < attackcount; akid ++) {
            host.attack[akid] = attackjson[akid].get<Real>();
        }

        falmErrCheckMacro(falmMemcpy(dev.r, host.r, sizeof(Real)*apcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.chord, host.chord, sizeof(Real)*apcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.twist, host.twist, sizeof(Real)*apcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.cl, host.cl, sizeof(Real)*apcount*attackcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.cd, host.cd, sizeof(Real)*apcount*attackcount, MCP::Hst2Dev));
        falmErrCheckMacro(falmMemcpy(dev.attack, host.attack, sizeof(Real)*attackcount, MCP::Hst2Dev));

        falmErrCheckMacro(falmMallocDevice((void**)&devptr, sizeof(APFrame)));
        falmErrCheckMacro(falmMemcpy(devptr, &dev, sizeof(APFrame), MCP::Hst2Dev));
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