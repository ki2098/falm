#pragma once

#include "typedef.h"

namespace Falm {

__host__ __device__ static REAL Upwind3(
    REAL phicc,
    REAL phie1, REAL phie2, REAL phiw1, REAL phiw2,
    REAL phin1, REAL phin2, REAL phis1, REAL phis2,
    REAL phit1, REAL phit2, REAL phib1, REAL phib2,
    REAL Ucc, REAL Vcc, REAL Wcc
) {
    REAL adv = 0;
    adv += (Ucc*(- phie2 + phie1 - phiw1 + phiw2) + 0.5*fabs(Ucc)*(phie2 - 4*phie1 + 6*phicc - 4*phiw1 + phiw2))/12;
    adv += (Vcc*(- phin2 + phin1 - phis1 + phis2) + 0.5*fabs(Vcc)*(phin2 - 4*phin1 + 6*phicc - 4*phis1 + phis2))/12;
    adv += (Wcc*(- phit2 + phit1 - phib1 + phib2) + 0.5*fabs(Wcc)*(phit2 - 4*phit1 + 6*phicc - 4*phib1 + phib2))/12;
    return adv;
}

}