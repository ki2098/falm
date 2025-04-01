#ifndef FALM_FALMCFDSCHEME_H
#define FALM_FALMCFDSCHEME_H

#include "typedef.h"

namespace Falm {

__host__ __device__ static inline Real UpwindFlux1st(Real ul, Real ur, Real UF) {
    return 0.5 * (UF * (ur + ul) - fabs(UF) * (ur - ul));
}

__host__ __device__ static Real Upwind1st(
    Real ucc,
    Real ue1, Real uw1,
    Real un1, Real us1,
    Real ut1, Real ub1,
    Real JUE, Real JUW,
    Real JVN, Real JVS,
    Real JWT, Real JWB,
    Real jacobian
) {
    Real adv = 0.0;
    adv += UpwindFlux1st(ucc, ue1, JUE);
    adv -= UpwindFlux1st(uw1, ucc, JUW);
    adv += UpwindFlux1st(ucc, un1, JVN);
    adv -= UpwindFlux1st(us1, ucc, JVS);
    adv += UpwindFlux1st(ucc, ut1, JWT);
    adv -= UpwindFlux1st(ub1, ucc, JWB);
    adv /= jacobian;
    return adv;
} 

__host__ __device__ static Real QUICK(
    Real ucc,
    Real ue1, Real ue2, Real uw1, Real uw2,
    Real un1, Real un2, Real us1, Real us2,
    Real ut1, Real ut2, Real ub1, Real ub2,
    Real JUE, Real JUW,
    Real JVN, Real JVS,
    Real JWT, Real JWB,
    Real jacobian
) {
    Real adv = 0.;

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

__host__ __device__ static Real KK(
    Real ucc,
    Real ue1, Real ue2, Real uw1, Real uw2,
    Real un1, Real un2, Real us1, Real us2,
    Real ut1, Real ut2, Real ub1, Real ub2,
    Real Uabs, Real Vabs, Real Wabs,
    Real JUE, Real JUW,
    Real JVN, Real JVS,
    Real JWT, Real JWB,
    Real jacobian
) {
    Real adv = 0.0;
    Real UDL, UDR, UD4;
    const Real ALPHA = 1./4.;

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

__host__ __device__ static Real UTOPIA(
    Real ucc,
    Real ue1, Real ue2, Real uw1, Real uw2,
    Real un1, Real un2, Real us1, Real us2,
    Real ut1, Real ut2, Real ub1, Real ub2,
    Real Uabs, Real Vabs, Real Wabs,
    Real JUE, Real JUW,
    Real JVN, Real JVS,
    Real JWT, Real JWB,
    Real jacobian
) {
    Real adv = 0.0;
    Real UDL, UDR, UD4;
    const Real ALPHA = 1./12.;

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

__host__ __device__ static Real Upwind3rd(
    Real ucc,
    Real ue1, Real ue2, Real uw1, Real uw2,
    Real un1, Real un2, Real us1, Real us2,
    Real ut1, Real ut2, Real ub1, Real ub2,
    Real Uabs, Real Vabs, Real Wabs,
    Real JUE, Real JUW,
    Real JVN, Real JVS,
    Real JWT, Real JWB,
    Real jacobian
) {
    Real adv = 0.0;
    Real UDL, UDR, UD4;
    const Real ALPHA = 1./24.;

    UDL = JUE*(- ue2 + 27*ue1 - 27*ucc + uw1)/24.;
    UDR = JUW*(- ue1 + 27*ucc - 27*uw1 + uw2)/24.;
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

__host__ __device__ static Real Diffusion(
    Real ReI,
    Real ucc,
    Real ue1, Real uw1,
    Real un1, Real us1,
    Real ut1, Real ub1,
    Real nutcc,
    Real nute1, Real nutw1,
    Real nutn1, Real nuts1,
    Real nutt1, Real nutb1,
    Real gxcc, Real gxe1, Real gxw1,
    Real gycc, Real gyn1, Real gys1,
    Real gzcc, Real gzt1, Real gzb1,
    Real jacobian
) {
    Real vis = 0.0;
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
