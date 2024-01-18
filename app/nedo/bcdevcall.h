#ifndef FALM_APP_TURBINEJSON_BCDEVCALL_H
#define FALM_APP_TURBINEJSON_BCDEVCALL_H

#include "../../src/matrix.h"
#include "../../src/region.h"

void ubc_xminus(Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void ubc_xplus (Falm::Matrix<Falm::REAL> &u, Falm::Matrix<Falm::REAL> &uprev, Falm::Matrix<Falm::REAL> &x, Falm::REAL dt, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void ubc_yminus(Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void ubc_yplus (Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void ubc_zminus(Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void ubc_zplus (Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);

void pbc_xminus(Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void pbc_xplus (Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void pbc_yminus(Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void pbc_yplus (Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void pbc_zminus(Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);
void pbc_zplus (Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM s);

#endif