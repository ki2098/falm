#ifndef FALM_ALM_ALMDEVCALL_H
#define FALM_ALM_ALMDEVCALL_H

#include "aparray.h"
#include "turbinearray.h"
#include "../CPMBase.h"

namespace Falm {

__host__ __device__ static inline REAL3 getcdcl(MatrixFrame<REAL> *vapc, size_t apid, REAL attack, REAL &cd, REAL &cl) {
    MatrixFrame<REAL> &apc = *vapc;
    size_t imax = apc.shape[0]-1;
    if (attack < apc(0,0)) {
        cd = apc(0,1);
        cl = apc(0,2);
    } else if (attack >= apc(imax,0)) {
        cd = apc(imax,1);
        cl = apc(imax,2);
    } else {
        for (size_t i = 0; i < imax; i ++) {
            REAL a0 = apc(i  ,0);
            REAL a1 = apc(i+1,0);
            if (attack >= a0 && attack < a1) {
                REAL p = (attack - a0)/(a1 - a0);
                REAL cd0 = apc(i  ,1);
                REAL cd1 = apc(i+1,1);
                REAL cl0 = apc(i  ,2);
                REAL cl1 = apc(i+1,2);
                cd = (1 - p)*cd0 + p*cd1;
                cl = (1 - p)*cl0 + p*cl1;
            }
        }
    }
}

__host__ __device__ static inline REAL3 pitch_only_vector_turbine2abs(const REAL3 &x, REAL pitch) {
    return REAL3{{
         x[0]*cos(pitch) + x[2]*sin(pitch),
         x[1],
        -x[0]*sin(pitch) + x[2]*cos(pitch)
    }};
}

__host__ __device__ static inline REAL3 pitch_only_velocity_abs2turbine(const REAL3 &u, const REAL3 &x, const REAL3 &x0, REAL pitch, REAL pitchRate) {
    REAL S = sin(pitch), C = cos(pitch);
    REAL U  =  u[0], V  =  u[1], W  =  u[2];
    REAL X  =  x[0], Y  =  x[1], Z  =  x[2];
    REAL X0 = x0[0], Y0 = x0[1], Z0 = x0[2];
    REAL XH = X-X0 , YH = Y-Y0 , ZH = Z-Z0 ;
    REAL &O = pitchRate;
    
    REAL _u = O*(- XH*S - ZH*C) + U*C - W*S;
    REAL _v = V;
    REAL _w = O*(  XH*C - ZH*S) + U*S + W*C;

    return REAL3{{_u, _v, _w}};
}

class FalmAlmDevCall {
public:
    Matrix<INT> ioffset, joffset, koffset;
    INT3 mpi_shape;
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
        Matrix<REAL> &x, Matrix<REAL> &y, Matrix<REAL> &z,
        Matrix<REAL> &u,
        REAL tt,
        CPM &cpm,
        int block_size = 32
    );

    void updateForce(
        APArray &aps,
        Matrix<REAL> &x,
        Matrix<REAL> &ff,
        REAL euler_eps,
        CPM &cpm,
        dim3 block_dim=dim3(8,8,8)
    );
};

}

#endif