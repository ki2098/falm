#ifndef FALM_APP_LD2D_BCDEVCALL_H
#define FALM_APP_LD2D_BCDEVCALL_H

#include "../../src/matrix.h"
#include "../../src/region.h"

void ubc_xminus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void ubc_xplus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void ubc_yminus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void ubc_yplus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);

void uubc_xminus(Falm::Matrix<Falm::Real> &uu, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void uubc_xplus(Falm::Matrix<Falm::Real> &uu, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void uubc_yminus(Falm::Matrix<Falm::Real> &uu, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void uubc_yplus(Falm::Matrix<Falm::Real> &uu, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);

void pbc_xminus(Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void pbc_xplus(Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void pbc_yminus(Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);
void pbc_yplus(Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream s);

void copy_z5(Falm::Matrix<Falm::Real> &v, Falm::Region &pdm, Falm::Int gc);

#endif