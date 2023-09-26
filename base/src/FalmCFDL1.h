#ifndef FALM_FALMCFDL1_H
#define FALM_FALMCFDL1_H

#include "matrix.h"
#include "mapper.h"

namespace Falm {

class L1Explicit {
public:
    double Re;
    double ReI;
    double dt;

    L1Explicit(double _Re, double _dt): Re(_Re), ReI(1 / _Re), dt(_dt) {}

    void L1Dev_Cartesian_FSCalcPseudoU(
        Matrix<double> &u,
        Matrix<double> &uu,
        Matrix<double> &ua,
        Matrix<double> &nut,
        Matrix<double> &kx,
        Matrix<double> &g,
        Matrix<double> &jac,
        Matrix<double> &ff,
        Mapper         &proc_domain,
        dim3            block_dim
    ) {
        Mapper map(proc_domain, Gd);
        L0Dev_Cartesian_FSCalcPseudoU(u, uu, ua, nut, kx, g, jac, ff, proc_domain, map, block_dim);
    }
    void L1Dev_Cartesian_UtoCU(
        Matrix<double> &u,
        Matrix<double> &uc,
        Matrix<double> &kx,
        Matrix<double> &jac,
        Mapper         &proc_domain,
        dim3            block_dim
    ) {
        Mapper map(proc_domain, Gd);
        L0Dev_Cartesian_UtoCU(u, uc, kx, jac, proc_domain, map, block_dim);
    }

protected:
    void L0Dev_Cartesian_FSCalcPseudoU(
        Matrix<double> &u,
        Matrix<double> &uu,
        Matrix<double> &ua,
        Matrix<double> &nut,
        Matrix<double> &kx,
        Matrix<double> &g,
        Matrix<double> &jac,
        Matrix<double> &ff,
        Mapper         &proc_domain,
        Mapper         &map,
        dim3            block_dim
    );
    void L0Dev_Cartesian_UtoCU(
        Matrix<double> &u,
        Matrix<double> &uc,
        Matrix<double> &kx,
        Matrix<double> &jac,
        Mapper         &proc_domain,
        Mapper         &map,
        dim3            block_dim
    );
    void L0Dev_Cartesian_InterpolateCU(

    );
    
};

}

#endif