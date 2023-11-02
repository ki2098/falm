#ifndef FALM_FALMCFDL2_H
#define FALM_FALMCFDL2_H

#include "FalmCFDDevCall.h"
#include "FalmEq.h"
#include "MV.h"

namespace Falm {

class FalmCFD : public FalmCFDDevCall {
public:
    FalmCFD(REAL _Re, REAL _dt, FLAG _AdvScheme, FLAG _SGSModel = SGSType::Smagorinsky, REAL _CSmagorinsky = 0.1) : 
        FalmCFDDevCall(_Re, _dt, _AdvScheme, _SGSModel, _CSmagorinsky) 
    {}

    void FSPseudoU(
        Matrix<REAL> &un,
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &ua,
        Matrix<REAL> &nut,
        Matrix<REAL> &kx,
        Matrix<REAL> &g,
        Matrix<REAL> &ja,
        Matrix<REAL> &ff,
        CPMBase      &cpm,
        dim3          block_dim,
        STREAM       *stream = nullptr
    );

    void UtoUU(
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        CPMBase      &cpm,
        dim3          block_dim,
        STREAM       *stream = nullptr
    );

    void ProjectP(
        Matrix<REAL> &u,
        Matrix<REAL> &ua,
        Matrix<REAL> &uu,
        Matrix<REAL> &uua,
        Matrix<REAL> &p,
        Matrix<REAL> &kx,
        Matrix<REAL> &g,
        CPMBase      &cpm,
        dim3          block_dim,
        STREAM       *stream = nullptr
    );

    void SGS(
        Matrix<REAL> &u,
        Matrix<REAL> &nut,
        Matrix<REAL> &x,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        CPMBase      &cpm,
        dim3          block_dim,
        STREAM       *stream = nullptr
    );

    void Divergence(
        Matrix<REAL> &uu,
        Matrix<REAL> &dvr,
        Matrix<REAL> &ja,
        CPMBase      &cpm,
        dim3          block_dim
    ) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region map(pdm.shape, cpm.gc);
        FalmCFDDevCall::Divergence(uu, dvr, ja, pdm, map, block_dim);
    }

    void MACCalcPoissonRHS(
        Matrix<REAL> &uu,
        Matrix<REAL> &rhs,
        Matrix<REAL> &ja,
        CPMBase      &cpm,
        dim3          block_dim,
        REAL          maxdiag = 1.0
    ) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region map(pdm.shape, cpm.gc);
        FalmCFDDevCall::Divergence(uu, rhs, ja, pdm, map, block_dim);
        MV::ScaleMatrix(rhs, 1.0 / (dt * maxdiag), block_dim);
    }
    
};



}

#endif