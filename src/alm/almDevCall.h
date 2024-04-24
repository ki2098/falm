#ifndef FALM_ALM_ALMDEVCALL_H
#define FALM_ALM_ALMDEVCALL_H

#include "../CPMBase.h"
#include "../matrix.h"

namespace Falm {

struct TurbineArray {
    int NT, NBPT, NAPPB;
    Matrix<REAL> x;
    Matrix<REAL> rcxT, R;
    Matrix<REAL> pitch, pitchRate, tipRate;

    void init(int _NT, int _NBPT, int _NAPPB) {
        NT = _NT;
        NBPT = _NBPT;
        NAPPB = _NAPPB;
        x.alloc(NT, 3, HDC::HstDev);
        rcxT.alloc(NT, 3, HDC::HstDev);
        R.alloc(NT, 1, HDC::HstDev);
        pitch.alloc(NT, 1, HDC::HstDev);
        pitchRate.alloc(NT, 1, HDC::HstDev);
        tipRate.alloc(NT, 1, HDC::HstDev);
    }

    void release() {
        x.release();
        rcxT.release();
        R.release();
        pitch.release();
        pitchRate.release();
        tipRate.release();
    }
};

class APArray {
public:
    Matrix<REAL> apx, apr, apTh;
    Matrix<REAL> apChord, apTwist;
    Matrix<REAL> apf;
    Matrix<int> turbineIdx;
    Matrix<int> aprank;
    Matrix<INT> aplocal;

    void init(int n_turbines, int n_blades_per_turbine, int n_aps_per_blade) {
        int nap = n_turbines * n_blades_per_turbine * n_aps_per_blade;
        apx.alloc(nap, 3, HDC::HstDev);
        apTh.alloc(nap, 3, HDC::HstDev);
        apr.alloc(nap, 1, HDC::HstDev);
        apChord.alloc(nap, 1, HDC::HstDev);
        apTwist.alloc(nap, 1, HDC::HstDev);
        apf.alloc(nap, 3, HDC::Device);
        turbineIdx.alloc(nap, 1, HDC::HstDev);
        aprank.alloc(nap, 1, HDC::HstDev);
        aplocal.alloc(nap, 3, HDC::Device);
        int n_ap_per_turbine = n_aps_per_blade * n_blades_per_turbine;
        for (int i = 0; i < nap; i ++) {
            turbineIdx(i) = i / n_ap_per_turbine;
        }
        turbineIdx.sync(MCP::Hst2Dev);
    }

    void release() {
        apx.release();
        apr.release();
        apTh.release();
        apChord.release();
        apTwist.release();
        apf.release();
        turbineIdx.release();
        aprank.release();
        aplocal.release();
    }
};

__host__ __device__ static inline REAL3 pitch_only_velocity_abs2turbine(const REAL3 &u, const REAL3 &x, const REAL3 &x0, REAL pitch, REAL pitchRate) {
    REAL S = sin(pitch), C = cos(pitch);
    REAL U  =  u[0], V  =  u[1], W  =  u[2];
    REAL X  =  x[0], Y  =  x[1], Z  =  x[2];
    REAL X0 = x0[0], Y0 = x0[1], Z0 = x0[2];
    REAL &O = pitchRate;
    
    REAL _u = O*(  X0*S + Z0*C - X*S - Z*C) + U*C - W*S;
    REAL _v = V;
    REAL _w = O*(- X0*C + Z0*C + X*C - Z*S) + U*S + W*C;

    return REAL3{{_u, _v, _w}};
}

__host__ __device__ static inline REAL3 pitch_only_vector_turbine2abs(const REAL3 &x, REAL pitch) {
    return REAL3{{
         x[0]*cos(pitch) + x[2]*sin(pitch),
         x[1],
        -x[0]*sin(pitch) + x[2]*cos(pitch)
    }};
}

class FalmAlmDevCall {
public:
    Matrix<INT> IOffset, JOffset, KOffset;
    INT3 mpi_shape;
    APArray ap_array;
    TurbineArray turbine_array;
public:
    void init(int n_turbines, int n_blades_per_turbine, int n_aps_per_blade, CPM &cpm) {
        turbine_array.init(n_turbines, n_blades_per_turbine, n_aps_per_blade);
        ap_array.init(n_turbines, n_blades_per_turbine, n_aps_per_blade);

        IOffset.alloc(cpm.shape[0], 1, HDC::HstDev);
        JOffset.alloc(cpm.shape[1], 1, HDC::HstDev);
        KOffset.alloc(cpm.shape[2], 1, HDC::HstDev);
        for (int i = 0; i < cpm.shape[0]; i ++) {
            IOffset(i) = cpm.pdm_list[IDX(i,0,0,cpm.shape)].offset[0] + cpm.gc - 1;
        }
        for (int j = 0; j < cpm.shape[1]; j ++) {
            JOffset(j) = cpm.pdm_list[IDX(0,j,0,cpm.shape)].offset[1] + cpm.gc - 1;
        }
        for (int k = 0; k < cpm.shape[2]; k ++) {
            KOffset(k) = cpm.pdm_list[IDX(0,0,k,cpm.shape)].offset[2] + cpm.gc - 1;
        }
    }

    void release() {
        ap_array.release();
        turbine_array.release();
        IOffset.release();
        JOffset.release();
        KOffset.release();
    }

    void CalcAPForce(APArray &ap, const TurbineArray &turbine, const Matrix<REAL> &u, const Matrix<REAL> &x, const Matrix<REAL> &y, const Matrix<REAL> &z, const CPM &cpm);

};

__host__ __device__ static inline REAL cdfunc(REAL attack) {
    
}

__host__ __device__ static inline REAL clfunc(REAL attack) {
    
}

}

#endif