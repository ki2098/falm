#ifndef FALM_ALM_TURBINEARRAY_H
#define FALM_ALM_TURBINEARRAY_H

#include "../matrix.h"

namespace Falm {

class TurbineArray {
public:
    int nt, nbpt, nappb;
    Matrix<Real> foundx;
    Matrix<Real> hubx;
    Matrix<Real> pitch;
    Matrix<Real> pitchrate;
    Matrix<Real> tiprate;
    Real r;

public:
    void init(int _nt, int _nbpt, int _nappb) {
        nt    = _nt;
        nbpt  = _nbpt;
        nappb = _nappb;
        foundx.alloc(nt, 3, HDC::HstDev);
        hubx.alloc(nt, 3, HDC::HstDev);
        pitch.alloc(nt, 1, HDC::HstDev);
        pitchrate.alloc(nt, 1, HDC::HstDev);
        tiprate.alloc(nt, 1, HDC::HstDev);
    }

    void release() {
        foundx.release();
        hubx.release();
        pitch.release();
        pitchrate.release();
        tiprate.release();
    }
};

}

#endif