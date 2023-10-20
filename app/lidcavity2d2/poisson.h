#ifndef _LID_CAVITY2D2_POISSON_H_
#define _LID_CAVITY2D2_POISSON_H_

#include "poissonDev.h"
#include "../../src/MVL2.h"

namespace LidCavity2d2 {

static Falm::REAL makePoissonMatrix(
    Falm::Matrix<Falm::REAL> &a,
    Falm::Matrix<Falm::REAL> &g,
    Falm::Matrix<Falm::REAL> &ja,
    Falm::CPMBase            &cpm,
    dim3                      block_dim = dim3(8, 8, 1)
) {
    Falm::Region &global = cpm.global;
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    dev_makePoissonMatrix(a, g, ja, global, pdm, cpm.gc, block_dim);
    Falm::REAL maxdiag = Falm::L2Dev_MaxDiag(a, block_dim, cpm);
    Falm::L1Dev_ScaleMatrix(a, 1.0 / maxdiag, block_dim);
    return maxdiag;
}

}

#endif