#pragma once

#include "typedef.h"

namespace Falm {

__host__ __device__ static Real Upwind3(
    Real phicc,
    Real phie1, Real phie2, Real phiw1, Real phiw2,
    Real phin1, Real phin2, Real phis1, Real phis2,
    Real phit1, Real phit2, Real phib1, Real phib2,
    Real Ucc, Real Vcc, Real Wcc
) {
    Real adv = 0;
    adv += (Ucc*(- phie2 + phie1 - phiw1 + phiw2) + 0.5*fabs(Ucc)*(phie2 - 4*phie1 + 6*phicc - 4*phiw1 + phiw2))/12;
    adv += (Vcc*(- phin2 + phin1 - phis1 + phis2) + 0.5*fabs(Vcc)*(phin2 - 4*phin1 + 6*phicc - 4*phis1 + phis2))/12;
    adv += (Wcc*(- phit2 + phit1 - phib1 + phib2) + 0.5*fabs(Wcc)*(phit2 - 4*phit1 + 6*phicc - 4*phib1 + phib2))/12;
    return adv;
}

}