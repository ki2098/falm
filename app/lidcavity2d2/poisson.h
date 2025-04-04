#ifndef _LID_CAVITY2D2_POISSON_H_
#define _LID_CAVITY2D2_POISSON_H_

#include "poissonDev.h"
#include "../../src/MV.h"

namespace LidCavity2d2 {

static Falm::Real makePoissonMatrix(
    Falm::Matrix<Falm::Real> &a,
    Falm::Matrix<Falm::Real> &g,
    Falm::Matrix<Falm::Real> &ja,
    Falm::CPM            &cpm,
    dim3                      block_dim = dim3(8, 8, 1)
) {
    Falm::Region &global = cpm.global;
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    dev_makePoissonMatrix(a, g, ja, global, pdm, cpm.gc, block_dim);
    Falm::Real maxdiag = Falm::FalmMV::MaxDiag(a, cpm, block_dim);
    Falm::FalmMV::ScaleMatrix(a, 1.0 / maxdiag, block_dim);
    return maxdiag;
}

}

#endif