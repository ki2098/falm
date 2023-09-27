#ifndef FALM_FALMCFDSCHEME_H
#define FALM_FALMCFDSCHEME_H

#include "typedef.h"

namespace Falm {

__host__ __device__ static inline REAL UpwindFlux1st(REAL ul, REAL ur, REAL UF) {
    return 0.5 * (UF * (ur + ul) - fabs(UF) * (ur - ul));
}

__host__ __device__ static REAL Upwind1st(
    REAL ucc,
    REAL ue1, REAL uw1,
    REAL un1, REAL us1,
    REAL ut1, REAL ub1,
    REAL UE, REAL UW,
    REAL VN, REAL VS,
    REAL WT, REAL WB,
    REAL jacobian
) {
    REAL adv = 0.0;
    adv += UpwindFlux1st(ucc, ue1, UE);
    adv -= UpwindFlux1st(uw1, ucc, UW);
    adv += UpwindFlux1st(ucc, un1, VN);
    adv -= UpwindFlux1st(us1, ucc, VS);
    adv += UpwindFlux1st(ucc, ut1, WT);
    adv -= UpwindFlux1st(ub1, ucc, WB);
    adv /= jacobian;
    return adv;
} 

__host__ __device__ static REAL Upwind3rd(
    REAL ucc,
    REAL ue1, REAL ue2, REAL uw1, REAL uw2,
    REAL un1, REAL un2, REAL us1, REAL us2,
    REAL ut1, REAL ut2, REAL ub1, REAL ub2,
    REAL Uabs, REAL Vabs, REAL Wabs,
    REAL UE, REAL UW,
    REAL VN, REAL VS,
    REAL WT, REAL WB,
    REAL jacobian
) {
    REAL adv = 0.0;
    REAL jx2 = 2 * jacobian;
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

__host__ __device__ static REAL Diffusion(
    REAL ReI,
    REAL ucc,
    REAL ue1, REAL uw1,
    REAL un1, REAL us1,
    REAL ut1, REAL ub1,
    REAL nutcc,
    REAL nute1, REAL nutw1,
    REAL nutn1, REAL nuts1,
    REAL nutt1, REAL nutb1,
    REAL gxcc, REAL gxe1, REAL gxw1,
    REAL gycc, REAL gyn1, REAL gys1,
    REAL gzcc, REAL gzt1, REAL gzb1,
    REAL jacobian
) {
    REAL vis = 0.0;
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