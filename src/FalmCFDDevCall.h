#ifndef FALM_FALMCFDL1_H
#define FALM_FALMCFDL1_H

#include "matrix.h"
#include "region.h"
#include "FalmEqDevCall.h"

namespace Falm {

class AdvectionSchemeType {
public:
    static const Flag Upwind1 = 0;
    static const Flag Upwind3 = 1;
    static const Flag QUICK   = 2;
    static const Flag UTOPIA  = 3;
    static const Flag KK      = 4;
};

class SGSType {
public:
    static const Flag Empty       = 0;
    static const Flag Smagorinsky = 1;
    static const Flag CSM         = 2;
};

class FalmCFDDevCall {
public:
    Real Re;
    Real ReI;
    // REAL dt;
    Flag AdvScheme;
    Flag SGSModel;
    Real CSmagorinsky;

    FalmCFDDevCall() {}
    FalmCFDDevCall(Real _Re, Flag _AdvScheme, Flag _SGSModel = SGSType::Smagorinsky, Real _CSmagorinsky = 0.1):
        Re(_Re),
        ReI(1.0 / _Re),
        AdvScheme(_AdvScheme),
        SGSModel(_SGSModel),
        CSmagorinsky(_CSmagorinsky) 
    {}

    void init(Real _Re, Flag _AdvScheme, Flag _SGSModel = SGSType::Smagorinsky, Real _CSmagorinsky = 0.1) {
        Re = _Re;
        ReI = 1.0 / _Re;
        AdvScheme = _AdvScheme;
        SGSModel = _SGSModel;
        CSmagorinsky = _CSmagorinsky;
    }

    // void L1Dev_Cartesian3d_FSCalcPseudoU(
    //     Matrix<REAL> &un,
    //     Matrix<REAL> &u,
    //     Matrix<REAL> &uu,
    //     Matrix<REAL> &ua,
    //     Matrix<REAL> &nut,
    //     Matrix<REAL> &kx,
    //     Matrix<REAL> &g,
    //     Matrix<REAL> &ja,
    //     Matrix<REAL> &ff,
    //     CPMBase      &cpm,
    //     dim3          block_dim
    // ) {
    //     Region &pdm = cpm.pdm_list[cpm.rank];
    //     Region map(pdm.shape, cpm.gc);
    //     FSPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, pdm, map, block_dim);
    // }

    // void L1Dev_Cartesian3d_UtoCU(
    //     Matrix<REAL> &u,
    //     Matrix<REAL> &uc,
    //     Matrix<REAL> &kx,
    //     Matrix<REAL> &ja,
    //     CPMBase      &cpm,
    //     dim3          block_dim
    // ) {
    //     Region &pdm = cpm.pdm_list[cpm.rank];
    //     Region map(pdm.shape, cpm.gc - 1);
    //     UtoCU(u, uc, kx, ja, pdm, map, block_dim);
    // }

    // void L1Dev_Cartesian3d_InterpolateCU(
    //     Matrix<REAL> &uu,
    //     Matrix<REAL> &uc,
    //     CPMBase      &cpm,
    //     dim3          block_dim
    // ) {
    //     Region &pdm = cpm.pdm_list[cpm.rank];
    //     Region map(pdm.shape, cpm.gc);
    //     map.shape  += {1, 1, 1};
    //     map.offset -= {1, 1, 1};
    //     // map = map.transform(
    //     //     INT3{ 1,  1,  1},
    //     //     INT3{-1, -1, -1}
    //     // );
    //     InterpolateCU(uu, uc, pdm, map, block_dim);
    // }

    // void L1Dev_Cartesian3d_ProjectPGrid(
    //     Matrix<REAL> &u,
    //     Matrix<REAL> &ua,
    //     Matrix<REAL> &p,
    //     Matrix<REAL> &kx,
    //     CPMBase      &cpm,
    //     dim3          block_dim
    // ) {
    //     Region &pdm = cpm.pdm_list[cpm.rank];
    //     Region map(pdm.shape, cpm.gc);
    //     ProjectPGrid(u, ua, p, kx, pdm, map, block_dim);
    // }

    // void L1Dev_Cartesian3d_ProjectPFace(
    //     Matrix<REAL> &uu,
    //     Matrix<REAL> &uua,
    //     Matrix<REAL> &p,
    //     Matrix<REAL> &g,
    //     CPMBase      &cpm,
    //     dim3          block_dim
    // ) {
    //     Region &pdm = cpm.pdm_list[cpm.rank];
    //     Region map(pdm.shape, cpm.gc);
    //     map.shape  += {1, 1, 1};
    //     map.offset -= {1, 1, 1};
    //     // map = map.transform(
    //     //     INT3{ 1,  1,  1},
    //     //     INT3{-1, -1, -1}
    //     // );
    //     ProjectPFace(uu, uua, p, g, pdm, map, block_dim);
    // }

    // void L1Dev_Cartesian3d_SGS(
    //     Matrix<REAL> &u,
    //     Matrix<REAL> &nut,
    //     Matrix<REAL> &x,
    //     Matrix<REAL> &kx,
    //     Matrix<REAL> &ja,
    //     CPMBase      &cpm,
    //     dim3          block_dim
    // ) {
    //     if (SGSModel == SGSType::Empty) {
    //         return;
    //     }
    //     Region &pdm = cpm.pdm_list[cpm.rank];
    //     Region map(pdm.shape, cpm.gc);
    //     SGS(u, nut, x, kx, ja, pdm, map, block_dim);
    // }

    // void L1Dev_Cartesian3d_Divergence(
    //     Matrix<REAL> &uu,
    //     Matrix<REAL> &div,
    //     Matrix<REAL> &ja,
    //     CPMBase      &cpm,
    //     dim3          block_dim
    // ) {
    //     Region &pdm = cpm.pdm_list[cpm.rank];
    //     Region map(pdm.shape, cpm.gc);
    //     Divergence(uu, div, ja, pdm, map, block_dim);
    // }

    // void L1Dev_Cartesian3d_MACCalcPoissonRHS(
    //     Matrix<REAL> &uu,
    //     Matrix<REAL> &rhs,
    //     Matrix<REAL> &ja,
    //     CPMBase      &cpm,
    //     dim3          block_dim,
    //     REAL          maxdiag = 1.0
    // ) {
    //     L1Dev_Cartesian3d_Divergence(uu, rhs, ja, cpm, block_dim);
    //     L1Dev_ScaleMatrix(rhs, 1.0 / (dt * maxdiag), block_dim);
    // }


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
        Region       &pdm,
        const Region &map,
        dim3          block_dim,
        Stream        stream = (Stream)0
    );
    void UtoCU(
        Matrix<Real> &u,
        Matrix<Real> &uc,
        Matrix<Real> &kx,
        Matrix<Real> &ja,
        Region       &pdm,
        const Region &map,
        dim3          block_dim,
        Stream        stream = (Stream)0
    );
    void InterpolateCU(
        Matrix<Real> &uu,
        Matrix<Real> &uc,
        Region       &pdm,
        const Region &map,
        dim3          block_dim,
        Stream        stream = (Stream)0
    );
    void ProjectPGrid(
        Matrix<Real> &u,
        Matrix<Real> &ua,
        Matrix<Real> &p,
        Matrix<Real> &kx,
        Real dt,
        Region       &pdm,
        const Region &map,
        dim3          block_dim,
        Stream        stream = (Stream)0
    );
    void ProjectPFace(
        Matrix<Real> &uu,
        Matrix<Real> &uua,
        Matrix<Real> &p,
        Matrix<Real> &g,
        Real dt,
        Region       &pdm,
        const Region &map,
        dim3          block_dim,
        Stream        stream = (Stream)0
    );
    void SGS(
        Matrix<Real> &u,
        Matrix<Real> &nut,
        Matrix<Real> &x,
        Matrix<Real> &kx,
        Matrix<Real> &ja,
        Region       &pdm,
        const Region &map,
        dim3          block_dim,
        Stream        stream = (Stream)0
    );
    void Divergence(
        Matrix<Real> &uu,
        Matrix<Real> &div,
        Matrix<Real> &ja,
        Region       &pdm,
        const Region &map,
        dim3          block_dim,
        Stream        stream = (Stream)0
    );
    void Vortcity(
        Matrix<Real> &u,
        Matrix<Real> &kx,
        Matrix<Real> &vrt,
        Region       &pdm,
        const Region &map,
        dim3          block_dim,
        Stream        stream = 0
    );
};

}

#endif
