#ifndef FALM_ALM_ALMDEVCALL_H
#define FALM_ALM_ALMDEVCALL_H

#include "aparray.h"
#include "turbinearray.h"
#include "../CPMBase.h"

namespace Falm {

__host__ __device__ static inline Real3 getcdcl(MatrixFrame<Real> *vapc, size_t apid, Real attack, Real &cd, Real &cl) {
    MatrixFrame<Real> &apc = *vapc;
    size_t imax = apc.shape[0]-1;
    if (attack < apc(0,0)) {
        cd = apc(0,1);
        cl = apc(0,2);
    } else if (attack >= apc(imax,0)) {
        cd = apc(imax,1);
        cl = apc(imax,2);
    } else {
        for (size_t i = 0; i < imax; i ++) {
            Real a0 = apc(i  ,0);
            Real a1 = apc(i+1,0);
            if (attack >= a0 && attack < a1) {
                Real p = (attack - a0)/(a1 - a0);
                Real cd0 = apc(i  ,1);
                Real cd1 = apc(i+1,1);
                Real cl0 = apc(i  ,2);
                Real cl1 = apc(i+1,2);
                cd = (1 - p)*cd0 + p*cd1;
                cl = (1 - p)*cl0 + p*cl1;
            }
        }
    }
}

__host__ __device__ static inline Real3 pitch_only_vector_turbine2abs(const Real3 &x, Real pitch) {
    return Real3{{
         x[0]*cos(pitch) + x[2]*sin(pitch),
         x[1],
        -x[0]*sin(pitch) + x[2]*cos(pitch)
    }};
}

__host__ __device__ static inline Real3 pitch_only_velocity_abs2turbine(const Real3 &u, const Real3 &x, const Real3 &x0, Real pitch, Real pitchRate) {
    Real S = sin(pitch), C = cos(pitch);
    Real U  =  u[0], V  =  u[1], W  =  u[2];
    Real X  =  x[0], Y  =  x[1], Z  =  x[2];
    Real X0 = x0[0], Y0 = x0[1], Z0 = x0[2];
    Real XH = X-X0 , YH = Y-Y0 , ZH = Z-Z0 ;
    Real &O = pitchRate;
    
    Real _u = O*(- XH*S - ZH*C) + U*C - W*S;
    Real _v = V;
    Real _w = O*(  XH*C - ZH*S) + U*S + W*C;

    return Real3{{_u, _v, _w}};
}

class FalmAlmDevCall {
public:
    Matrix<Int> ioffset, joffset, koffset;
    Int3 mpi_shape;
    APArray aps;
    TurbineArray turbines;

public:
    void init(int nt, int nbpt, int nappb, CPM &cpm) {
        turbines.init(nt, nbpt, nappb);
        aps.init(nt, nbpt, nappb);

        ioffset.alloc(cpm.shape[0]+1, 1, HDC::HstDev);
        joffset.alloc(cpm.shape[1]+1, 1, HDC::HstDev);
        koffset.alloc(cpm.shape[2]+1, 1, HDC::HstDev);
        for (int i = 0; i < cpm.shape[0]; i ++) {
            ioffset(i) = cpm.pdoffset[0][i] + cpm.gc - 1;
        }
        for (int j = 0; j < cpm.shape[1]; j ++) {
            joffset(j) = cpm.pdoffset[1][j] + cpm.gc - 1;
        }
        for (int k = 0; k < cpm.shape[2]; k ++) {
            koffset(k) = cpm.pdoffset[2][k] + cpm.gc - 1;
        }
        mpi_shape = cpm.shape;
        ioffset.sync(MCP::Hst2Dev);
        joffset.sync(MCP::Hst2Dev);
        koffset.sync(MCP::Hst2Dev);
    }

    void release() {
        ioffset.release();
        joffset.release();
        koffset.release();
        aps.release();
        turbines.release();
    }

    void updateAp(
        APArray &aps,
        TurbineArray &turbines,
        Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z,
        Matrix<Real> &u,
        Real tt,
        CPM &cpm,
        int block_size = 32
    );

    void updateForce(
        APArray &aps,
        Matrix<Real> &x,
        Matrix<Real> &ff,
        Real euler_eps,
        CPM &cpm,
        dim3 block_dim=dim3(8,8,8)
    );
};

}

#endif