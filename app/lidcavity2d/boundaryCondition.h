#ifndef _LID_CAVITY2D_BOUNDARY_CONDITION_H_
#define _LID_CAVITY2D_BOUNDARY_CONDITION_H_

#include "../../src/matrix.h"
#include "../../src/mapper.h"

namespace LidCavity2d {

void pressureBC(
    Falm::Matrix<Falm::REAL> &p,
    Falm::Mapper             &pdm,
    Falm::STREAM             *streamptr = nullptr
);

void velocityBC(
    Falm::Matrix<Falm::REAL> &u,
    Falm::Mapper             &pdm,
    Falm::STREAM             *streamptr = nullptr
);

void copyZ5(
    Falm::Matrix<Falm::REAL> &field,
    Falm::Mapper             &pdm,
    Falm::STREAM             *streamptr = nullptr
);

// static inline void forceZComponentZero(Falm::Matrix<Falm::REAL> &field, Falm::FLAG hdctype) {
//     assert(field.shape.y == 3);
//     if (hdctype & Falm::HDCType::Host) {
//         assert(field.hdctype & Falm::HDCType::Host);
//         Falm::falmMemset(&field(0, 2), 0, sizeof(Falm::REAL) * field.shape.x);
//     }
//     if (hdctype & Falm::HDCType::Device) {
//         assert(field.hdctype & Falm::HDCType::Device);
//         Falm::falmMemsetDevice(&field.dev(0, 2), 0, sizeof(Falm::REAL) * field.shape.x);
//     }
// }

}

#endif