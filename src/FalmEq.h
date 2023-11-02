#ifndef FALM_STRUCTEQL2_H
#define FALM_STRUCTEQL2_H

#include "FalmEqDevCall.h"
#include "CPM.h"

namespace Falm {

// void L2Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr);



class FalmEq : public FalmEqDevCall {
public:
    FalmEq(FLAG _type, INT _maxit, REAL _tol, REAL _relax_factor, FLAG _pc_type = SolverType::Empty, INT _pc_maxit = 5, REAL _pc_relax_factor = 1.0) : 
        FalmEqDevCall(_type, _maxit, _tol, _relax_factor, _pc_type, _pc_maxit, _pc_relax_factor) 
    {}

    void Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void SOR(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void Solve(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr) {
        if (type == SolverType::Jacobi) {
            Jacobi(a, x, b, r, cpm, block_dim, stream);
        } else if (type == SolverType::SOR) {
            SOR(a, x, b, r, cpm, block_dim, stream);
        } else if (type == SolverType::PBiCGStab) {
            PBiCGStab(a, x, b, r, cpm, block_dim, stream);
        }
    }

public:
    static void Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void JacobiPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void SORPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void Precondition(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr) {
        if (pc_type == SolverType::Jacobi) {
            JacobiPC(a, x, b, cpm, block_dim, stream);
        } else if (pc_type == SolverType::SOR) {
            SORPC(a, x, b, cpm, block_dim, stream);
        }
    }

};

}

#endif
