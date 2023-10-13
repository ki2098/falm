#ifndef _LID_CAVITY2D2_POISSONDEV_H_
#define _LID_CAVITY2D2_POISSONDEV_H_

#include "../../src/matrix.h"
#include "../../src/mapper.h"

namespace LidCavity2d2 {

void dev_makePoissonMatrix(
    Falm::Matrix<Falm::REAL> &a,
    Falm::Matrix<Falm::REAL> &g,
    Falm::Matrix<Falm::REAL> &ja,
    Falm::Mapper             &global,
    Falm::Mapper             &pdm,
    dim3                      block_dim = dim3(8, 8, 1)
);

}

#endif