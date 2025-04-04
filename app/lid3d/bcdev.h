#ifndef FALM_APP_LID3D_BCDEV_H
#define FALM_APP_LID3D_BCDEV_H

#include "../../src/matrix.h"
#include "../../src/region.h"

namespace LID3D {

void ubc_xminus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void ubc_xplus (Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void ubc_yminus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void ubc_yplus (Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void ubc_zminus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void ubc_zplus (Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);


void pbc_xminus(Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void pbc_xplus (Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void pbc_yminus(Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void pbc_yplus (Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void pbc_zminus(Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void pbc_zplus (Falm::Matrix<Falm::Real> &p, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);

void uubc_xminus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void uubc_xplus (Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void uubc_yminus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void uubc_yplus (Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void uubc_zminus(Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);
void uubc_zplus (Falm::Matrix<Falm::Real> &u, Falm::Region &pdm, Falm::Int gc, Falm::Stream stream);

}

#endif