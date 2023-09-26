#ifndef FALM_FALMCFDL1_H
#define FALM_FALMCFDL1_H

#include "matrix.h"
#include "mapper.h"
#include "structEqL1.h"

namespace Falm {

class L1Explicit {
public:
    REAL Re;
    REAL ReI;
    REAL dt;

    L1Explicit(REAL _Re, REAL _dt): Re(_Re), ReI(1 / _Re), dt(_dt) {}

    void L1Dev_Cartesian_FSCalcPseudoU(
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &ua,
        Matrix<REAL> &nut,
        Matrix<REAL> &kx,
        Matrix<REAL> &g,
        Matrix<REAL> &jac,
        Matrix<REAL> &ff,
        Mapper       &proc_domain,
        dim3          block_dim
    ) {
        Mapper map(proc_domain, Gd);
        L0Dev_Cartesian_FSCalcPseudoU(u, uu, ua, nut, kx, g, jac, ff, proc_domain, map, block_dim);
    }

    void L1Dev_Cartesian_UtoCU(
        Matrix<REAL> &u,
        Matrix<REAL> &uc,
        Matrix<REAL> &kx,
        Matrix<REAL> &jac,
        Mapper       &proc_domain,
        dim3          block_dim
    ) {
        Mapper map(proc_domain, Gd);
        L0Dev_Cartesian_UtoCU(u, uc, kx, jac, proc_domain, map, block_dim);
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

protected:
    void L0Dev_Cartesian_FSCalcPseudoU(
        Matrix<REAL> &u,
        Matrix<REAL> &uu,
        Matrix<REAL> &ua,
        Matrix<REAL> &nut,
        Matrix<REAL> &kx,
        Matrix<REAL> &g,
        Matrix<REAL> &jac,
        Matrix<REAL> &ff,
        Mapper       &proc_domain,
        Mapper       &map,
        dim3          block_dim
    );
    void L0Dev_Cartesian_UtoCU(
        Matrix<REAL> &u,
        Matrix<REAL> &uc,
        Matrix<REAL> &kx,
        Matrix<REAL> &jac,
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
    
};

}

#endif