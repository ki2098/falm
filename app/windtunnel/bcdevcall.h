#pragma once

#include "../../src/matrix.h"
#include "../../src/region.h"

void ubc_xminus(Falm::Matrix<Falm::Real> &u, Falm::Real u_inlet, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void ubc_xplus (Falm::Matrix<Falm::Real> &u, Falm::Matrix<Falm::Real> &uprev, Falm::Matrix<Falm::Real> &x, Falm::Real dt, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void ubc_yminus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void ubc_yplus (Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void ubc_zminus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void ubc_zplus (Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);

void pbc_xminus(Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void pbc_xplus (Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void pbc_yminus(Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void pbc_yplus (Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void pbc_zminus(Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void pbc_zplus (Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);