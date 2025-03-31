#ifndef FALM_FALMCFDL2_H
#define FALM_FALMCFDL2_H

#include <string>
#include "FalmCFDDevCall.h"
#include "FalmEq.h"
#include "MV.h"

namespace Falm {

class FalmCFD : public FalmCFDDevCall {
public:
    Matrix<Real> uc;

    FalmCFD() {}
    FalmCFD(Real _Re, Flag _AdvScheme, Flag _SGSModel = SGSType::Smagorinsky, Real _CSmagorinsky = 0.1) : 
        FalmCFDDevCall(_Re, _AdvScheme, _SGSModel, _CSmagorinsky) 
    {}

    void init(Real _Re, Flag _AdvScheme, Flag _SGSModel = SGSType::Smagorinsky, Real _CSmagorinsky = 0.1) {
        FalmCFDDevCall::init(_Re, _AdvScheme, _SGSModel, _CSmagorinsky);
    }

    void alloc(Int3 shape) {
        uc.alloc(shape, 3, HDC::Device, "contra U at control volume centers");
    }

    void release() {
        uc.release();
    }

    void FSPseudoU(
        Matrix<Real> &un,
        Matrix<Real> &u,
        Matrix<Real> &uu,
        Matrix<Real> &ua,
        Matrix<Real> &nut,
        Matrix<Real> &kx,
        Matrix<Real> &g,
        Matrix<Real> &ja,
        Matrix<Real> &ff,
        Real dt,
        CPM      &cpm,
        dim3          block_dim,
        Stream       *stream = nullptr,
        Int           margin = 0
    );

    void UtoUU(
        Matrix<Real> &u,
        Matrix<Real> &uu,
        Matrix<Real> &kx,
        Matrix<Real> &ja,
        CPM      &cpm,
        dim3          block_dim,
        Stream       *stream = nullptr
    );

    void ProjectP(
        Matrix<Real> &u,
        Matrix<Real> &ua,
        Matrix<Real> &uu,
        Matrix<Real> &uua,
        Matrix<Real> &p,
        Matrix<Real> &kx,
        Matrix<Real> &g,
        Real dt,
        CPM      &cpm,
        dim3          block_dim,
        Stream       *stream = nullptr
    );

    void SGS(
        Matrix<Real> &u,
        Matrix<Real> &nut,
        Matrix<Real> &x,
        Matrix<Real> &kx,
        Matrix<Real> &ja,
        CPM      &cpm,
        dim3          block_dim,
        Stream       *stream = nullptr
    );

    void Divergence(
        Matrix<Real> &uu,
        Matrix<Real> &dvr,
        Matrix<Real> &ja,
        CPM      &cpm,
        dim3          block_dim
    ) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region map(pdm.shape, cpm.gc);
        FalmCFDDevCall::Divergence(uu, dvr, ja, pdm, map, block_dim);
    }

    void MACCalcPoissonRHS(
        Matrix<Real> &uu,
        Matrix<Real> &rhs,
        Matrix<Real> &ja,
        Real dt,
        CPM      &cpm,
        dim3          block_dim,
        Real          maxdiag = 1.0
    ) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region map(pdm.shape, cpm.gc);
        FalmCFDDevCall::Divergence(uu, rhs, ja, pdm, map, block_dim);
        FalmMV::ScaleMatrix(rhs, 1.0 / (dt * maxdiag), block_dim);
    }
    
    static Flag str2advscheme(std::string str) {
        if (str == "Upwind1") {
            return AdvectionSchemeType::Upwind1;
        } else if (str == "Upwind3") {
            return AdvectionSchemeType::Upwind3;
        } else if (str == "QUICK") {
            return AdvectionSchemeType::QUICK;
        } else if (str == "UTOPIA") {
            return AdvectionSchemeType::UTOPIA;
        } else if (str == "KK") {
            return AdvectionSchemeType::KK;
        } else {
            return AdvectionSchemeType::Upwind1;
        }
    }

    static Flag str2sgs(std::string str) {
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

    static std::string advscheme2str(Flag adv) {
        if (adv == AdvectionSchemeType::Upwind1) {
            return "Upwind1";
        } else if (adv == AdvectionSchemeType::Upwind3) {
            return "Upwind3";
        } else if (adv == AdvectionSchemeType::QUICK) {
            return "QUICK";
        } else if (adv == AdvectionSchemeType::UTOPIA) {
            return "UTOPIA";
        } else if (adv == AdvectionSchemeType::KK) {
            return "KK";
        } else {
            return "Not defined";
        }
    }

    static std::string sgs2str(Flag sgs) {
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
