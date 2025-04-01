#ifndef _LID_CAVITY2D_POISSON_H_
#define _LID_CAVITY2D_POISSON_H_

#include "../../src/matrix.h"
#include "../../src/CPMBase.h"

namespace LidCavity2d {

Falm::Real makePoissonMatrix(
    Falm::Matrix<Falm::Real> &a,
    Falm::Matrix<Falm::Real> &g,
    Falm::Matrix<Falm::Real> &ja,
    Falm::CPM            &cpm,
    dim3                      block_dim = dim3(8, 8, 1)
);

void makePoissonRHS(
    Falm::Matrix<Falm::Real> &p,
    Falm::Matrix<Falm::Real> &rhs,
    Falm::Matrix<Falm::Real> &g,
    Falm::Matrix<Falm::Real> &ja,
    Falm::Real                maxdiag,
    Falm::CPM            &cpm,
    dim3                      block_dim = dim3(8, 8, 1)
);

}

#endif