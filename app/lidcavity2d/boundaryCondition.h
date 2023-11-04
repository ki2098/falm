#ifndef _LID_CAVITY2D_BOUNDARY_CONDITION_H_
#define _LID_CAVITY2D_BOUNDARY_CONDITION_H_

#include "../../src/matrix.h"
#include "../../src/CPMBase.h"

namespace LidCavity2d {

void pressureBC(
    Falm::Matrix<Falm::REAL> &p,
    Falm::CPMBase            &cpm,
    Falm::STREAM             *streamptr = nullptr
);

void velocityBC(
    Falm::Matrix<Falm::REAL> &u,
    Falm::CPMBase            &cpm,
    Falm::STREAM             *streamptr = nullptr
);

void forceFaceVelocityZero(
    Falm::Matrix<Falm::REAL> &uu,
    Falm::CPMBase            &cpm,
    Falm::STREAM             *streamptr = nullptr
);

void copyZ5(
    Falm::Matrix<Falm::REAL> &field,
    Falm::CPMBase            &cpm,
    Falm::STREAM             *streamptr = nullptr
);

// static inline void forceZComponentZero(Falm::Matrix<Falm::REAL> &field, Falm::FLAG hdctype) {
//     assert(field.shape[1] == 3);
//     if (hdctype & Falm::HDCType::Host) {
//         assert(field.hdctype & Falm::HDCType::Host);
//         Falm::falmMemset(&field(0, 2), 0, sizeof(Falm::REAL) * field.shape[0]);
//     }
//     if (hdctype & Falm::HDCType::Device) {
//         assert(field.hdctype & Falm::HDCType::Device);
//         Falm::falmMemsetDevice(&field.dev(0, 2), 0, sizeof(Falm::REAL) * field.shape[0]);
//     }
// }

}

#endif