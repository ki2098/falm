#ifndef FALM_STRUCTEQL1_H
#define FALM_STRUCTEQL1_H

#include "matrix.h"
#include "CPMBase.h"

namespace Falm {

// void MVDevCall::MVMult(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Region &pdm, const Region &map, dim3 block_dim, STREAM stream = (STREAM)0);



// static inline void L1Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     MVDevCall::MVMult(a, x, ax, pdm, map, block_dim);
// }

// static inline void L1Dev_Struct3d7p_Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     L0Dev_Struct3d7p_Res(a, x, b, r, pdm, map, block_dim);
// }

class FalmEqDevCall {
public:
    INT            maxit;
    REAL             tol;
    INT               it;
    REAL             err;
    REAL    relax_factor;
    INT         pc_maxit;
    REAL pc_relax_factor;

    class SolverType {
    public:
        static const FLAG Empty     = 0;
        static const FLAG Jacobi    = 1;
        static const FLAG SOR       = 2;
        static const FLAG PBiCGStab = 4;
    };

    FLAG    type;
    FLAG pc_type;

    FalmEqDevCall(FLAG _type, INT _maxit, REAL _tol, REAL _relax_factor, FLAG _pc_type = SolverType::Empty, INT _pc_maxit = 5, REAL _pc_relax_factor = 1.0) : 
        type(_type), maxit(_maxit), tol(_tol), relax_factor(_relax_factor),
        pc_type(_pc_type), pc_maxit(_pc_maxit), pc_relax_factor(_pc_relax_factor) 
    {}

    // void L1Dev_Struct3d7p_Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim);
    // void L1Dev_Struct3d7p_SOR(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim);
    // void L1Dev_Struct3d7p_PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim);
    // void L1Dev_Struct3d7p_Solve(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim) {
    //     if (type == SolverType::Jacobi) {
    //         L1Dev_Struct3d7p_Jacobi(a, x, b, r, cpm, block_dim);
    //     } else if (type == SolverType::SOR) {
    //         L1Dev_Struct3d7p_SOR(a, x, b, r, cpm, block_dim);
    //     } else if (type == SolverType::PBiCGStab) {
    //         L1Dev_Struct3d7p_PBiCGStab(a, x, b, r, cpm, block_dim);
    //     }
    // }

public:
    static void Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Region &pdm, const Region &map, dim3 block_dim, STREAM stream = (STREAM)0);
    void PBiCGStab1(Matrix<REAL> &p, Matrix<REAL> &q, Matrix<REAL> &r, REAL beta, REAL omega, Region &pdm, const Region &map, dim3 block_dim);
    void PBiCGStab2(Matrix<REAL> &s, Matrix<REAL> &q, Matrix<REAL> &r, REAL alpha, Region &pdm, const Region &map, dim3 block_dim);
    void PBiCGStab3(Matrix<REAL> &x, Matrix<REAL> &pp, Matrix<REAL> &ss, REAL alpha, REAL omega, Region &pdm, const Region &map, dim3 block_dim);
    void PBiCGStab4(Matrix<REAL> &r, Matrix<REAL> &s, Matrix<REAL> &t, REAL omega, Region &pdm, const Region &map, dim3 block_dim);
    void JacobiSweep(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &xp, Matrix<REAL> &b, Region &pdm, const Region &map, dim3 block_dim, STREAM stream = (STREAM)0);
    void SORSweep(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, REAL omega, INT color, Region &pdm, const Region &map, dim3 block_dim, STREAM stream = (STREAM)0);
    // void L1Dev_Struct3d7p_JacobiPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim);
    // void L1Dev_Struct3d7p_SORPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim);
    // void L1Dev_Struct3d7p_Precondition(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim) {
    //     if (pc_type == SolverType::Jacobi) {
    //         L1Dev_Struct3d7p_JacobiPC(a, x, b, cpm, block_dim);
    //     } else if (pc_type == SolverType::SOR) {
    //         L1Dev_Struct3d7p_SORPC(a, x, b, cpm, block_dim);
    //     }
    // }



};

typedef FalmEqDevCall::SolverType LSType;

}

#endif
