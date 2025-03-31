#ifndef FALM_ALM_ALMDEVCALL_H
#define FALM_ALM_ALMDEVCALL_H

#include "../CPMBase.h"
#include "../matrix.h"

namespace Falm {

struct TurbineArray {
    int NTurbine, NBladePerTurbine, NAPPerBlade;
    Matrix<Real> x;
    Matrix<Real> rcxT, R;
    Matrix<Real> pitch, pitchRate, tipRate;

    void init(int _NT, int _NBPT, int _NAPPB) {
        NTurbine = _NT;
        NBladePerTurbine = _NBPT;
        NAPPerBlade = _NAPPB;
        x.alloc(NTurbine, 3, HDC::HstDev);
        rcxT.alloc(NTurbine, 3, HDC::HstDev);
        R.alloc(NTurbine, 1, HDC::HstDev);
        pitch.alloc(NTurbine, 1, HDC::HstDev);
        pitchRate.alloc(NTurbine, 1, HDC::HstDev);
        tipRate.alloc(NTurbine, 1, HDC::HstDev);
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
    Matrix<Real> apx, apr, apTh;
    Matrix<Real> apChord, apTwist;
    Matrix<Real> apf;
    Matrix<int> turbineIdx;
    Matrix<int> aprank;
    Matrix<Int> aplocal;

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

__host__ __device__ static inline Real3 pitch_only_velocity_abs2turbine(const Real3 &u, const Real3 &x, const Real3 &x0, Real pitch, Real pitchRate) {
    Real S = sin(pitch), C = cos(pitch);
    Real U  =  u[0], V  =  u[1], W  =  u[2];
    Real X  =  x[0], Y  =  x[1], Z  =  x[2];
    Real X0 = x0[0], Y0 = x0[1], Z0 = x0[2];
    Real &O = pitchRate;
    
    Real _u = O*(  X0*S + Z0*C - X*S - Z*C) + U*C - W*S;
    Real _v = V;
    Real _w = O*(- X0*C + Z0*C + X*C - Z*S) + U*S + W*C;

    return Real3{{_u, _v, _w}};
}

__host__ __device__ static inline Real3 pitch_only_vector_turbine2abs(const Real3 &x, Real pitch) {
    return Real3{{
         x[0]*cos(pitch) + x[2]*sin(pitch),
         x[1],
        -x[0]*sin(pitch) + x[2]*cos(pitch)
    }};
}

class FalmAlmDevCall {
public:
    Matrix<Int> IOffset, JOffset, KOffset;
    Int3 mpi_shape;
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
        IOffset.sync(MCP::Hst2Dev);
        JOffset.sync(MCP::Hst2Dev);
        KOffset.sync(MCP::Hst2Dev);
        mpi_shape = cpm.shape;
    }

    void release() {
        ap_array.release();
        turbine_array.release();
        IOffset.release();
        JOffset.release();
        KOffset.release();
    }

    void CalcAPForce(APArray &ap, const TurbineArray &turbine, const Matrix<Real> &u, const Matrix<Real> &x, const Matrix<Real> &y, const Matrix<Real> &z, const CPM &cpm);

};

__host__ __device__ static inline Real cdfunc(Real attack) {
    
}

__host__ __device__ static inline Real clfunc(Real attack) {
    
}

}

#endif