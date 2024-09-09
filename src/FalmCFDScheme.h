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
    REAL JUE, REAL JUW,
    REAL JVN, REAL JVS,
    REAL JWT, REAL JWB,
    REAL jacobian
) {
    REAL adv = 0.0;
    adv += UpwindFlux1st(ucc, ue1, JUE);
    adv -= UpwindFlux1st(uw1, ucc, JUW);
    adv += UpwindFlux1st(ucc, un1, JVN);
    adv -= UpwindFlux1st(us1, ucc, JVS);
    adv += UpwindFlux1st(ucc, ut1, JWT);
    adv -= UpwindFlux1st(ub1, ucc, JWB);
    adv /= jacobian;
    return adv;
} 

__host__ __device__ static REAL QUICK(
    REAL ucc,
    REAL ue1, REAL ue2, REAL uw1, REAL uw2,
    REAL un1, REAL un2, REAL us1, REAL us2,
    REAL ut1, REAL ut2, REAL ub1, REAL ub2,
    REAL JUE, REAL JUW,
    REAL JVN, REAL JVS,
    REAL JWT, REAL JWB,
    REAL jacobian
) {
    REAL adv = 0.;

    adv += JUE*(- ue2 + 9*ue1 + 9*ucc - uw1);
    adv += fabs(JUE)*(ue2 - 3*ue1 + 3*ucc - uw1);

    adv -= JUW*(- ue1 + 9*ucc + 9*uw1 - uw2);
    adv -= fabs(JUW)*(ue1 - 3*ucc + 3*uw1 - uw2);

    adv += JVN*(- un2 + 9*un1 + 9*ucc - us1);
    adv += fabs(JVN)*(un2 - 3*un1 + 3*ucc - us1);

    adv -= JVS*(- un1 + 9*ucc + 9*us1 - us2);
    adv -= fabs(JVS)*(un1 - 3*ucc + 3*us1 - us2);

    adv += JWT*(- ut2 + 9*ut1 + 9*ucc - ub1);
    adv += fabs(JWT)*(ut2 - 3*ut1 + 3*ucc - ub1);

    adv -= JWB*(- ut1 + 9*ucc + 9*ub1 - ub2);
    adv -= fabs(JWB)*(ut1 - 3*ucc + 3*ub1 - ub2);

    adv /= (16*jacobian);

    return adv;
}

__host__ __device__ static REAL KK(
    REAL ucc,
    REAL ue1, REAL ue2, REAL uw1, REAL uw2,
    REAL un1, REAL un2, REAL us1, REAL us2,
    REAL ut1, REAL ut2, REAL ub1, REAL ub2,
    REAL Uabs, REAL Vabs, REAL Wabs,
    REAL JUE, REAL JUW,
    REAL JVN, REAL JVS,
    REAL JWT, REAL JWB,
    REAL jacobian
) {
    REAL adv = 0.0;
    REAL UDL, UDR, UD4;
    const REAL ALPHA = 1./4.;

    UDL = JUE*(- ue2 + 27*ue1 - 27*ucc + uw1)/24.;
    UDR = JUW*(- ue1 + 27*ucc - 27*uw1 + uw2)/24;
    UD4 = Uabs*(ue2 - 4*ue1 + 6*ucc - 4*uw1 + uw2)*ALPHA;
    adv += .5*(UDL + UDR)/jacobian + UD4;

    UDL = JVN*(- un2 + 27*un1 - 27*ucc + us1)/24.;
    UDR = JVS*(- un1 + 27*ucc - 27*us1 + us2)/24.;
    UD4 = Vabs*(un2 - 4*un1 + 6*ucc - 4*us1 + us2)*ALPHA;
    adv += .5*(UDL + UDR)/jacobian + UD4;

    UDL = JWT*(- ut2 + 27*ut1 - 27*ucc + ub1)/24.;
    UDR = JWB*(- ut1 + 27*ucc - 27*ub1 + ub2)/24.;
    UD4 = Wabs*(ut2 - 4*ut1 + 6*ucc - 4*ub1 + ub2)*ALPHA;
    adv += .5*(UDL + UDR)/jacobian + UD4;

    return adv;
}

__host__ __device__ static REAL UTOPIA(
    REAL ucc,
    REAL ue1, REAL ue2, REAL uw1, REAL uw2,
    REAL un1, REAL un2, REAL us1, REAL us2,
    REAL ut1, REAL ut2, REAL ub1, REAL ub2,
    REAL Uabs, REAL Vabs, REAL Wabs,
    REAL JUE, REAL JUW,
    REAL JVN, REAL JVS,
    REAL JWT, REAL JWB,
    REAL jacobian
) {
    REAL adv = 0.0;
    REAL UDL, UDR, UD4;
    const REAL ALPHA = 1./12.;

    UDL = JUE*(- ue2 + 27*ue1 - 27*ucc + uw1)/24.;
    UDR = JUW*(- ue1 + 27*ucc - 27*uw1 + uw2)/24;
    UD4 = Uabs*(ue2 - 4*ue1 + 6*ucc - 4*uw1 + uw2)*ALPHA;
    adv += .5*(UDL + UDR)/jacobian + UD4;

    UDL = JVN*(- un2 + 27*un1 - 27*ucc + us1)/24.;
    UDR = JVS*(- un1 + 27*ucc - 27*us1 + us2)/24.;
    UD4 = Vabs*(un2 - 4*un1 + 6*ucc - 4*us1 + us2)*ALPHA;
    adv += .5*(UDL + UDR)/jacobian + UD4;

    UDL = JWT*(- ut2 + 27*ut1 - 27*ucc + ub1)/24.;
    UDR = JWB*(- ut1 + 27*ucc - 27*ub1 + ub2)/24.;
    UD4 = Wabs*(ut2 - 4*ut1 + 6*ucc - 4*ub1 + ub2)*ALPHA;
    adv += .5*(UDL + UDR)/jacobian + UD4;

    return adv;
}

__host__ __device__ static REAL Upwind3rd(
    REAL ucc,
    REAL ue1, REAL ue2, REAL uw1, REAL uw2,
    REAL un1, REAL un2, REAL us1, REAL us2,
    REAL ut1, REAL ut2, REAL ub1, REAL ub2,
    REAL Uabs, REAL Vabs, REAL Wabs,
    REAL JUE, REAL JUW,
    REAL JVN, REAL JVS,
    REAL JWT, REAL JWB,
    REAL jacobian
) {
    REAL adv = 0.0;
    REAL UDL, UDR, UD4;
    const REAL ALPHA = 1./24.;

    UDL = JUE*(- ue2 + 27*ue1 - 27*ucc + uw1)/24.;
    UDR = JUW*(- ue1 + 27*ucc - 27*uw1 + uw2)/24;
    UD4 = Uabs*(ue2 - 4*ue1 + 6*ucc - 4*uw1 + uw2)*ALPHA;
    adv += .5*(UDL + UDR)/jacobian + UD4;

    UDL = JVN*(- un2 + 27*un1 - 27*ucc + us1)/24.;
    UDR = JVS*(- un1 + 27*ucc - 27*us1 + us2)/24.;
    UD4 = Vabs*(un2 - 4*un1 + 6*ucc - 4*us1 + us2)*ALPHA;
    adv += .5*(UDL + UDR)/jacobian + UD4;

    UDL = JWT*(- ut2 + 27*ut1 - 27*ucc + ub1)/24.;
    UDR = JWB*(- ut1 + 27*ucc - 27*ub1 + ub2)/24.;
    UD4 = Wabs*(ut2 - 4*ut1 + 6*ucc - 4*ub1 + ub2)*ALPHA;
    adv += .5*(UDL + UDR)/jacobian + UD4;

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
