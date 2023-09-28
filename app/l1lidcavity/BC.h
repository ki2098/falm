#ifndef FALMAPP_L1LIDCAVITY_BC_H
#define FALMAPP_L1LIDCAVITY_BC_H

#include "../../src/matrix.h"
#include "../../src/mapper.h"

typedef Falm::Matrix<Falm::REAL>      RealField;
typedef Falm::MatrixFrame<Falm::REAL> RealFrame;

void dev_pressureBC(RealField &p, Falm::Mapper &proc_domain);
void dev_velocityBC(RealField &u, Falm::Mapper &proc_domain);

#endif