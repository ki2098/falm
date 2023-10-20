#ifndef FALM_FALMCFDL2_H
#define FALM_FALMCFDL2_H

#include "FalmCFDL1.h"
#include "structEqL2.h"
#include "MVL2.h"

namespace Falm {

class L2CFD : public L1CFD {
public:
    L2CFD(REAL _Re, REAL _dt, FLAG _AdvScheme, FLAG _SGSModel = SGSType::Smagorinsky, REAL _CSmagorinsky = 0.1) : 
        L1CFD(_Re, _dt, _AdvScheme, _SGSModel, _CSmagorinsky) 
    {}

    void L2Dev_Cartesian3d_FSCalcPseudoU(
        Matrix<REAL> &un,
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &ua,
        Matrix<REAL> &nut,
        Matrix<REAL> &kx,
        Matrix<REAL> &g,
        Matrix<REAL> &ja,
        Matrix<REAL> &ff,
        dim3          block_dim,
        CPMBase      &cpm,
        STREAM       *stream = nullptr
    );

    void L2Dev_Cartesian3d_UtoUU(
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        dim3          block_dim,
        CPMBase      &cpm,
        STREAM       *stream = nullptr
    );

    void L2Dev_Cartesian3d_ProjectP(
        Matrix<REAL> &u,
        Matrix<REAL> &ua,
        Matrix<REAL> &uu,
        Matrix<REAL> &uua,
        Matrix<REAL> &p,
        Matrix<REAL> &kx,
        Matrix<REAL> &g,
        dim3          block_dim,
        CPMBase      &cpm,
        STREAM       *stream = nullptr
    );

    void L2Dev_Cartesian3d_SGS(
        Matrix<REAL> &u,
        Matrix<REAL> &nut,
        Matrix<REAL> &x,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        dim3          block_dim,
        CPMBase      &cpm,
        STREAM       *stream = nullptr
    );

    void L2Dev_Cartesian3d_Divergence(
        Matrix<REAL> &uu,
        Matrix<REAL> &dvr,
        Matrix<REAL> &ja,
        CPMBase      &cpm,
        dim3          block_dim
    ) {
        L1Dev_Cartesian3d_Divergence(uu, dvr, ja, cpm.pdm_list[cpm.rank], cpm.gc, block_dim);
    }

    void L2Dev_Cartesian3d_MACCalcPoissonRHS(
        Matrix<REAL> &uu,
        Matrix<REAL> &rhs,
        Matrix<REAL> &ja,
        CPMBase      &cpm,
        dim3          block_dim,
        REAL          maxdiag = 1.0
    ) {
        L1Dev_Cartesian3d_MACCalcPoissonRHS(uu, rhs, ja, cpm.pdm_list[cpm.rank], cpm.gc, block_dim, maxdiag);
    }
    
};



}

#endif