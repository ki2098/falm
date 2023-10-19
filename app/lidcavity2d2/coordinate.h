#ifndef _LID_CAVITY2D2_COORDINATE_H_
#define _LID_CAVITY2D2_COORDINATE_H_

#include "../../src/matrix.h"
#include "../../src/mapper.h"

namespace LidCavity2d2 {

void setCoord(
    Falm::REAL                side_lenth,
    Falm::INT                 side_n_cell,
    Falm::Mapper             &pdm,
    Falm::INT                 gc,
    Falm::Matrix<Falm::REAL> &x,
    Falm::Matrix<Falm::REAL> &h,
    Falm::Matrix<Falm::REAL> &kx,
    Falm::Matrix<Falm::REAL> &g,
    Falm::Matrix<Falm::REAL> &ja,
    dim3                      block_dim = dim3{8, 8, 1}
);

}

#endif