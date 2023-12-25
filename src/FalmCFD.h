#ifndef FALM_FALMCFDL2_H
#define FALM_FALMCFDL2_H

#include <string>
#include "FalmCFDDevCall.h"
#include "FalmEq.h"
#include "MV.h"

namespace Falm {

class FalmCFD : public FalmCFDDevCall {
public:
    FalmCFD() {}
    FalmCFD(REAL _Re, FLAG _AdvScheme, FLAG _SGSModel = SGSType::Smagorinsky, REAL _CSmagorinsky = 0.1) : 
        FalmCFDDevCall(_Re, _AdvScheme, _SGSModel, _CSmagorinsky) 
    {}

    void init(REAL _Re, FLAG _AdvScheme, FLAG _SGSModel = SGSType::Smagorinsky, REAL _CSmagorinsky = 0.1) {
        FalmCFDDevCall::init(_Re, _AdvScheme, _SGSModel, _CSmagorinsky);
    }

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
        REAL dt,
        CPM      &cpm,
        dim3          block_dim,
        STREAM       *stream = nullptr,
        INT           margin = 0
    );

    void UtoUU(
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        CPM      &cpm,
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
        REAL dt,
        CPM      &cpm,
        dim3          block_dim,
        STREAM       *stream = nullptr
    );

    void SGS(
        Matrix<REAL> &u,
        Matrix<REAL> &nut,
        Matrix<REAL> &x,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        CPM      &cpm,
        dim3          block_dim,
        STREAM       *stream = nullptr
    );

    void Divergence(
        Matrix<REAL> &uu,
        Matrix<REAL> &dvr,
        Matrix<REAL> &ja,
        CPM      &cpm,
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
        REAL dt,
        CPM      &cpm,
        dim3          block_dim,
        REAL          maxdiag = 1.0
    ) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region map(pdm.shape, cpm.gc);
        FalmCFDDevCall::Divergence(uu, rhs, ja, pdm, map, block_dim);
        FalmMV::ScaleMatrix(rhs, 1.0 / (dt * maxdiag), block_dim);
    }
    
    static FLAG str2advscheme(std::string str) {
        if (str == "Upwind1") {
            return AdvectionSchemeType::Upwind1;
        } else if (str == "Upwind3") {
            return AdvectionSchemeType::Upwind3;
        } else {
            return AdvectionSchemeType::Upwind1;
        }
    }

    static FLAG str2sgs(std::string str) {
        if (str == "Empty") {
            return SGSType::Empty;
        } else if (str == "Smagorinsky") {
            return SGSType::Smagorinsky;
        } else if (str == "CSM") {
            return SGSType::CSM;
        } else {
            return SGSType::Empty;
        }
    }

    static std::string advscheme2str(FLAG adv) {
        if (adv == AdvectionSchemeType::Upwind1) {
            return "Upwind1";
        } else if (adv == AdvectionSchemeType::Upwind3) {
            return "Upwind3";
        } else {
            return "Not defined";
        }
    }

    static std::string sgs2str(FLAG sgs) {
        if (sgs == SGSType::Empty) {
            return "Empty";
        } else if (sgs == SGSType::Smagorinsky) {
            return "Smagorinsky";
        } else if (sgs == SGSType::CSM) {
            return "CSM";
        } else {
            return "Not defined";
        }
    }
};



}

#endif
