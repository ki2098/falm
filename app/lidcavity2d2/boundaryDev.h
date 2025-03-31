#ifndef _LID_CAVITY2D2_BOUNDARYDEV_H_
#define _LID_CAVITY2D2_BOUNDARYDEV_H_

#include "../../src/matrix.h"
#include "../../src/region.h"

namespace LidCavity2d2 {

void dev_pressureBC_E(
    Falm::Matrix<Falm::Real> &p,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream = (Falm::Stream)0
);

void dev_pressureBC_W(
    Falm::Matrix<Falm::Real> &p,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream = (Falm::Stream)0
);

void dev_pressureBC_N(
    Falm::Matrix<Falm::Real> &p,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream = (Falm::Stream)0
);

void dev_pressureBC_S(
    Falm::Matrix<Falm::Real> &p,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream = (Falm::Stream)0
);

void dev_velocityBC_E(
    Falm::Matrix<Falm::Real> &u,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream
);

void dev_velocityBC_W(
    Falm::Matrix<Falm::Real> &u,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream
);

void dev_velocityBC_N(
    Falm::Matrix<Falm::Real> &u,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream
);

void dev_velocityBC_S(
    Falm::Matrix<Falm::Real> &u,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream
);

void dev_forceFaceVelocityZero_E(
    Falm::Matrix<Falm::Real> &uu,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream
);

void dev_forceFaceVelocityZero_W(
    Falm::Matrix<Falm::Real> &uu,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream
);

void dev_forceFaceVelocityZero_N(
    Falm::Matrix<Falm::Real> &uu,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream
);

void dev_forceFaceVelocityZero_S(
    Falm::Matrix<Falm::Real> &uu,
    Falm::Region             &pdm,
    Falm::Int                 gc,
    Falm::Stream              stream
);

}

#endif