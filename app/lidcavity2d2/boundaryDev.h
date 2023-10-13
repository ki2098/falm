#ifndef _LID_CAVITY2D2_BOUNDARYDEV_H_
#define _LID_CAVITY2D2_BOUNDARYDEV_H_

#include "../../src/matrix.h"
#include "../../src/mapper.h"

namespace LidCavity2d2 {

void dev_pressureBC_E(
    Falm::Matrix<Falm::REAL> &p,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream = (Falm::STREAM)0
);

void dev_pressureBC_W(
    Falm::Matrix<Falm::REAL> &p,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream = (Falm::STREAM)0
);

void dev_pressureBC_N(
    Falm::Matrix<Falm::REAL> &p,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream = (Falm::STREAM)0
);

void dev_pressureBC_S(
    Falm::Matrix<Falm::REAL> &p,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream = (Falm::STREAM)0
);

void dev_velocityBC_E(
    Falm::Matrix<Falm::REAL> &u,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream
);

void dev_velocityBC_W(
    Falm::Matrix<Falm::REAL> &u,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream
);

void dev_velocityBC_N(
    Falm::Matrix<Falm::REAL> &u,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream
);

void dev_velocityBC_S(
    Falm::Matrix<Falm::REAL> &u,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream
);

void dev_forceFaceVelocityZero_E(
    Falm::Matrix<Falm::REAL> &uu,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream
);

void dev_forceFaceVelocityZero_W(
    Falm::Matrix<Falm::REAL> &uu,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream
);

void dev_forceFaceVelocityZero_N(
    Falm::Matrix<Falm::REAL> &uu,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream
);

void dev_forceFaceVelocityZero_S(
    Falm::Matrix<Falm::REAL> &uu,
    Falm::Mapper             &pdm,
    Falm::STREAM              stream
);

}

#endif