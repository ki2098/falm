#ifndef FALM_APP_LD2D_BCDEVCALL_H
#define FALM_APP_LD2D_BCDEVCALL_H

#include "../../src/matrix.h"
#include "../../src/region.h"

void ubc_xminus(Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void ubc_xplus(Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void ubc_yminus(Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void ubc_yplus(Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);

void uubc_xminus(Falm::Matrix<Falm::REAL> &uu, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void uubc_xplus(Falm::Matrix<Falm::REAL> &uu, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void uubc_yminus(Falm::Matrix<Falm::REAL> &uu, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void uubc_yplus(Falm::Matrix<Falm::REAL> &uu, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);

void pbc_xminus(Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void pbc_xplus(Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void pbc_yminus(Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void pbc_yplus(Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);

void copy_z5(Falm::Matrix<Falm::REAL> &v, Falm::Region &pdm, Falm::INT gc);

#endif