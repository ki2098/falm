#ifndef _LID_CAVITY2D2_BOUNDARYDEV_H_
#define _LID_CAVITY2D2_BOUNDARYDEV_H_

#include "../../src/matrix.h"
#include "../../src/region.h"

namespace TURBINE1 {

void vbc_xminus(Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);
void vbc_xplus (Falm::Matrix<Falm::REAL> &u, Falm::Matrix<Falm::REAL> &un, Falm::Matrix<Falm::REAL> &x, Falm::REAL dt, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);
void vbc_yminus(Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);
void vbc_yplus (Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);
void vbc_zminus(Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);
void vbc_zplus (Falm::Matrix<Falm::REAL> &u, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);

void pbc_xminus(Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);
void pbc_xplus (Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);
void pbc_yminus(Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);
void pbc_yplus (Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);
void pbc_zminus(Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);
void pbc_zplus (Falm::Matrix<Falm::REAL> &p, Falm::Region &pdm, Falm::INT gc, Falm::STREAM stream);

}

#endif