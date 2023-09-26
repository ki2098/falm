#ifndef FALM_FALMCFDSCHEME_H
#define FALM_FALMCFDSCHEME_H

#include "typedef.h"

namespace Falm {

__host__ __device__ static double Riam3rdUpwind(
    double ucc,
    double ue1, double ue2, double uw1, double uw2,
    double un1, double un2, double us1, double us2,
    double ut1, double ut2, double ub1, double ub2,
    double Uabs, double Vabs, double Wabs,
    double UE, double UW,
    double VN, double VS,
    double WT, double WB,
    double jacobian
) {
    double adv = 0.0;
    double jx2 = 2 * jacobian;
    adv += UE * (- ue2 + 27 * ue1 - 27 * ucc + uw1) / jx2;
    adv += UW * (- ue1 + 27 * ucc - 27 * uw1 + uw2) / jx2;
    adv += Uabs * (ue2 - 4 * ue1 + 6 * ucc - 4 * uw1 + uw2);
    adv += VN * (- un2 + 27 * un1 - 27 * ucc + us1) / jx2;
    adv += VS * (- un1 + 27 * ucc - 27 * us1 + us2) / jx2;
    adv += Vabs * (un2 - 4 * un1 + 6 * ucc - 4 * us1 + us2);
    adv += WT * (- ut2 + 27 * ut1 - 27 * ucc + ub1) / jx2;
    adv += WB * (- ut1 + 27 * ucc - 27 * ub1 + ub2) / jx2;
    adv += Wabs * (ut2 - 4 * ut1 + 6 * ucc - 4 * ub1 + ub2);
    adv /= 24.0;
    return adv;
}

__host__ __device__ static double Diffusion(
    double ReI,
    double ucc,
    double ue1, double uw1,
    double un1, double us1,
    double ut1, double ub1,
    double nutcc,
    double nute1, double nutw1,
    double nutn1, double nuts1,
    double nutt1, double nutb1,
    double gxcc, double gxe1, double gxw1,
    double gycc, double gyn1, double gys1,
    double gzcc, double gzt1, double gzb1,
    double jacobian
) {
    double vis = 0.0;
    vis += (gxcc + gxe1) * (ReI + 0.5 * (nutcc + nute1)) * (ue1 - ucc);
    vis -= (gxcc + gxw1) * (ReI + 0.5 * (nutcc + nutw1)) * (ucc - uw1);
    vis += (gycc + gyn1) * (ReI + 0.5 * (nutcc + nutn1)) * (un1 - ucc);
    vis -= (gycc + gys1) * (ReI + 0.5 * (nutcc + nuts1)) * (ucc - us1);
    vis += (gzcc + gzt1) * (ReI + 0.5 * (nutcc + nutt1)) * (ut1 - ucc);
    vis -= (gzcc + gzb1) * (ReI + 0.5 * (nutcc + nutb1)) * (ucc - ub1);
    vis /= (2 * jacobian);
    return vis;
}

}

#endif