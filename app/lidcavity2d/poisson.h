#ifndef _LID_CAVITY2D_POISSON_H_
#define _LID_CAVITY2D_POISSON_H_

#include "../../src/matrix.h"
#include "../../src/CPMBase.h"

namespace LidCavity2d {

Falm::REAL makePoissonMatrix(
    Falm::Matrix<Falm::REAL> &a,
    Falm::Matrix<Falm::REAL> &g,
    Falm::Matrix<Falm::REAL> &ja,
    Falm::CPM            &cpm,
    dim3                      block_dim = dim3(8, 8, 1)
);

void makePoissonRHS(
    Falm::Matrix<Falm::REAL> &p,
    Falm::Matrix<Falm::REAL> &rhs,
    Falm::Matrix<Falm::REAL> &g,
    Falm::Matrix<Falm::REAL> &ja,
    Falm::REAL                maxdiag,
    Falm::CPM            &cpm,
    dim3                      block_dim = dim3(8, 8, 1)
);

}

#endif