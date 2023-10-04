#ifndef FALM_FALMCFDL1_H
#define FALM_FALMCFDL1_H

#include "matrix.h"
#include "mapper.h"
#include "structEqL1.h"
#include "MVL1.h"

namespace Falm {

class AdvectionSchemeType {
public:
    static const FLAG Upwind1 = 0;
    static const FLAG Upwind3 = 1;
};

class SGSType {
public:
    static const FLAG Empty       = 0;
    static const FLAG Smagorinsky = 1;
    static const FLAG CSM         = 2;
};

class L1CFD {
public:
    REAL Re;
    REAL ReI;
    REAL dt;
    FLAG AdvScheme;
    FLAG SGSModel;
    REAL CSmagorinsky;

    L1CFD(REAL _Re, REAL _dt, FLAG _AdvScheme, FLAG _SGSModel = SGSType::Smagorinsky, REAL _CSmagorinsky = 0.1):
        Re(_Re),
        ReI(1.0 / _Re),
        dt(_dt),
        AdvScheme(_AdvScheme),
        SGSModel(_SGSModel),
        CSmagorinsky(_CSmagorinsky) 
    {}

    void L1Dev_Cartesian3d_FSCalcPseudoU(
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &ua,
        Matrix<REAL> &nut,
        Matrix<REAL> &kx,
        Matrix<REAL> &g,
        Matrix<REAL> &ja,
        Matrix<REAL> &ff,
        Mapper       &pdm,
        dim3          block_dim
    ) {
        Mapper map(pdm, Gd);
        L0Dev_Cartesian3d_FSCalcPseudoU(u, uu, ua, nut, kx, g, ja, ff, pdm, map, block_dim);
    }

    void L1Dev_Cartesian3d_UtoCU(
        Matrix<REAL> &u,
        Matrix<REAL> &uc,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        Mapper       &pdm,
        dim3          block_dim
    ) {
        Mapper map(pdm, Gd - 1);
        L0Dev_Cartesian3d_UtoCU(u, uc, kx, ja, pdm, map, block_dim);
    }

    void L1Dev_Cartesian3d_InterpolateCU(
        Matrix<REAL> &uu,
        Matrix<REAL> &uc,
        Mapper       &pdm,
        dim3          block_dim
    ) {
        Mapper map(pdm, Gd);
        map = map.transform(
            INTx3{ 1,  1,  1},
            INTx3{-1, -1, -1}
        );
        L0Dev_Cartesian3d_InterpolateCU(uu, uc, pdm, map, block_dim);
    }

    void L1Dev_Cartesian3d_ProjectPGrid(
        Matrix<REAL> &u,
        Matrix<REAL> &ua,
        Matrix<REAL> &p,
        Matrix<REAL> &kx,
        Mapper       &pdm,
        dim3          block_dim
    ) {
        Mapper map(pdm, Gd);
        L0Dev_Cartesian3d_ProjectPGrid(u, ua, p, kx, pdm, map, block_dim);
    }

    void L1Dev_Cartesian3d_ProjectPFace(
        Matrix<REAL> &uu,
        Matrix<REAL> &uua,
        Matrix<REAL> &p,
        Matrix<REAL> &g,
        Mapper       &pdm,
        dim3          block_dim
    ) {
        Mapper map(pdm, Gd);
        map = map.transform(
            INTx3{ 1,  1,  1},
            INTx3{-1, -1, -1}
        );
        L0Dev_Cartesian3d_ProjectPFace(uu, uua, p, g, pdm, map, block_dim);
    }

    void L1Dev_Cartesian3d_SGS(
        Matrix<REAL> &u,
        Matrix<REAL> &nut,
        Matrix<REAL> &x,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        Mapper       &pdm,
        dim3          block_dim
    ) {
        if (SGSModel == SGSType::Empty) {
            return;
        }
        Mapper map(pdm, Gd);
        L0Dev_Cartesian3d_SGS(u, nut, x, kx, ja, pdm, map, block_dim);
    }

    void L1Dev_Cartesian3d_Divergence(
        Matrix<REAL> &uu,
        Matrix<REAL> &div,
        Matrix<REAL> &ja,
        Mapper       &pdm,
        dim3          block_dim
    ) {
        Mapper map(pdm, Gd);
        L0Dev_Cartesian3d_Divergence(uu, div, ja, pdm, map, block_dim);
    }

    void L1Dev_Cartesian3d_MACCalcPoissonRHS(
        Matrix<REAL> &uu,
        Matrix<REAL> &rhs,
        Matrix<REAL> &ja,
        Mapper       &pdm,
        dim3          block_dim,
        REAL          maxdiag = 1.0
    ) {
        L1Dev_Cartesian3d_Divergence(uu, rhs, ja, pdm, block_dim);
        L1Dev_ScaleMatrix(rhs, 1.0 / (dt * maxdiag), block_dim);
    }

protected:
    void L0Dev_Cartesian3d_FSCalcPseudoU(
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &ua,
        Matrix<REAL> &nut,
        Matrix<REAL> &kx,
        Matrix<REAL> &g,
        Matrix<REAL> &ja,
        Matrix<REAL> &ff,
        Mapper       &pdm,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian3d_UtoCU(
        Matrix<REAL> &u,
        Matrix<REAL> &uc,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        Mapper       &pdm,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian3d_InterpolateCU(
        Matrix<REAL> &uu,
        Matrix<REAL> &uc,
        Mapper       &pdm,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian3d_ProjectPGrid(
        Matrix<REAL> &u,
        Matrix<REAL> &ua,
        Matrix<REAL> &p,
        Matrix<REAL> &kx,
        Mapper       &pdm,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian3d_ProjectPFace(
        Matrix<REAL> &uu,
        Matrix<REAL> &uua,
        Matrix<REAL> &p,
        Matrix<REAL> &g,
        Mapper       &pdm,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian3d_SGS(
        Matrix<REAL> &u,
        Matrix<REAL> &nut,
        Matrix<REAL> &x,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        Mapper       &pdm,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian3d_Divergence(
        Matrix<REAL> &uu,
        Matrix<REAL> &div,
        Matrix<REAL> &ja,
        Mapper       &pdm,
        Mapper       &map,
        dim3          block_dim
    );
};

}

#endif