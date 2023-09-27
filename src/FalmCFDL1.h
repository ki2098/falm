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

    void L1Dev_Cartesian_FSCalcPseudoU(
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &ua,
        Matrix<REAL> &nut,
        Matrix<REAL> &kx,
        Matrix<REAL> &g,
        Matrix<REAL> &ja,
        Matrix<REAL> &ff,
        Mapper       &proc_domain,
        dim3          block_dim
    ) {
        Mapper map(proc_domain, Gd);
        L0Dev_Cartesian_FSCalcPseudoU(u, uu, ua, nut, kx, g, ja, ff, proc_domain, map, block_dim);
    }

    void L1Dev_Cartesian_UtoCU(
        Matrix<REAL> &u,
        Matrix<REAL> &uc,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        Mapper       &proc_domain,
        dim3          block_dim
    ) {
        Mapper map(proc_domain, Gd);
        L0Dev_Cartesian_UtoCU(u, uc, kx, ja, proc_domain, map, block_dim);
    }

    void L1Dev_Cartesian_InterpolateCU(
        Matrix<REAL> &uu,
        Matrix<REAL> &uc,
        Mapper       &proc_domain,
        dim3          block_dim
    ) {
        Mapper map(proc_domain, Gd);
        map = map.transform(
            INTx3{ 1,  1,  1},
            INTx3{-1, -1, -1}
        );
        L0Dev_Cartesian_InterpolateCU(uu, uc, proc_domain, map, block_dim);
    }

    void L1Dev_Cartesian_ProjectPGrid(
        Matrix<REAL> &u,
        Matrix<REAL> &ua,
        Matrix<REAL> &p,
        Matrix<REAL> &kx,
        Mapper       &proc_domain,
        dim3          block_dim
    ) {
        Mapper map(proc_domain, Gd);
        L0Dev_Cartesian_ProjectPGrid(u, ua, p, kx, proc_domain, map, block_dim);
    }

    void L1Dev_Cartesian_SGS(
        Matrix<REAL> &u,
        Matrix<REAL> &nut,
        Matrix<REAL> &x,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        Mapper       &proc_domain,
        dim3          block_dim
    ) {
        if (SGSModel == SGSType::Empty) {
            return;
        }
        Mapper map(proc_domain, Gd);
        L0Dev_Cartesian_SGS(u, nut, x, kx, ja, proc_domain, map, block_dim);
    }

    void L1Dev_Cartesian_Divergence(
        Matrix<REAL> &uu,
        Matrix<REAL> &div,
        Matrix<REAL> &ja,
        Mapper       &proc_domain,
        dim3          block_dim
    ) {
        Mapper map(proc_domain, Gd);
        L0Dev_Cartesian_Divergence(uu, div, ja, proc_domain, map, block_dim);
    }

    void L1Dev_Cartesian_MACCalcPoissonRHS(
        Matrix<REAL> &uu,
        Matrix<REAL> &rhs,
        Matrix<REAL> &ja,
        Mapper       &proc_domain,
        dim3          block_dim
    ) {
        L1Dev_Cartesian_Divergence(uu, rhs, ja, proc_domain, block_dim);
        L1Dev_ScaleMatrix(rhs, 1.0 / dt, block_dim);
    }

protected:
    void L0Dev_Cartesian_FSCalcPseudoU(
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &ua,
        Matrix<REAL> &nut,
        Matrix<REAL> &kx,
        Matrix<REAL> &g,
        Matrix<REAL> &ja,
        Matrix<REAL> &ff,
        Mapper       &proc_domain,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian_UtoCU(
        Matrix<REAL> &u,
        Matrix<REAL> &uc,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        Mapper       &proc_domain,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian_InterpolateCU(
        Matrix<REAL> &uu,
        Matrix<REAL> &uc,
        Mapper       &proc_domain,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian_ProjectPGrid(
        Matrix<REAL> &u,
        Matrix<REAL> &ua,
        Matrix<REAL> &p,
        Matrix<REAL> &kx,
        Mapper       &proc_domain,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian_ProjectPFace(
        Matrix<REAL> &uu,
        Matrix<REAL> &uua,
        Matrix<REAL> &p,
        Matrix<REAL> &g,
        Mapper       &proc_domain,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian_SGS(
        Matrix<REAL> &u,
        Matrix<REAL> &nut,
        Matrix<REAL> &x,
        Matrix<REAL> &kx,
        Matrix<REAL> &ja,
        Mapper       &proc_domain,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian_Divergence(
        Matrix<REAL> &uu,
        Matrix<REAL> &div,
        Matrix<REAL> &ja,
        Mapper       &proc_domain,
        Mapper       &map,
        dim3          block_dim
    );
};

}

#endif