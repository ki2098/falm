#ifndef _LID_CAVITY2D_COORDINATE_H_
#define _LID_CAVITY2D_COORDINATE_H_

#include "../../src/matrix.h"
#include "../../src/CPMBase.h"

namespace LidCavity2d {

void setCoord(
    Falm::Real                side_lenth,
    Falm::Int                 side_n_cell,
    Falm::CPM            &cpm,
    Falm::Matrix<Falm::Real> &x,
    Falm::Matrix<Falm::Real> &h,
    Falm::Matrix<Falm::Real> &kx,
    Falm::Matrix<Falm::Real> &g,
    Falm::Matrix<Falm::Real> &ja,
    dim3                      block_dim = dim3{8, 8, 1}
);

}

#endif