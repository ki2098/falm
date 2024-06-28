#ifndef FALM_ALM_APARRAY_H
#define FALM_ALM_APARRAY_H

#include "../matrix.h"

namespace Falm {

class APArray {
public:
    int nap;
    Matrix<REAL> apx;
    Matrix<INT>  api;
    Matrix<REAL> aptheta;
    Matrix<REAL> apr;
    Matrix<REAL> apchord;
    Matrix<REAL> aptwist;
    Matrix<REAL> apff;
    Matrix<REAL> apcdcl;
    Matrix<int> aptid;
    Matrix<int> apbid;
    Matrix<int> aprank;

public:
    void init(int nt, int nbpt, int nappb) {
        nap = nt*nbpt*nappb;
        apx.alloc(nap, 3, HDC::HstDev);
        api.alloc(nap, 3, HDC::HstDev);
        aptheta.alloc(nap, 1, HDC::HstDev);
        apr.alloc(nap, 1, HDC::HstDev);
        apchord.alloc(nap, 1, HDC::HstDev);
        aptwist.alloc(nap, 1, HDC::HstDev);
        apff.alloc(nap, 3, HDC::HstDev);
        aptid.alloc(nap, 1, HDC::HstDev);
        apbid.alloc(nap, 1, HDC::HstDev);
        aprank.alloc(nap, 1, HDC::HstDev);
        int nappt = nappb*nbpt;
        for (int i = 0; i < nap; i ++) {
            aptid(i) = i/nappt;
            apbid(i) = (i%nappt)/nappb;
        }
        aptid.sync(MCP::Hst2Dev);
        apbid.sync(MCP::Hst2Dev);
    }

    void init_cdcl() {

    }

    void release() {
        apx.release();
        api.release();
        aptheta.release();
        apr.release();
        apchord.release();
        aptwist.release();
        apff.release();
        apcdcl.release();
        aptid.release();
        apbid.release();
        aprank.release();
    }
};

}

#endif