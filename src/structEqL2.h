#ifndef FALM_STRUCTEQL2_H
#define FALM_STRUCTEQL2_H

#include "structEqL1.h"
#include "CPM.h"

namespace Falm {

void L2Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream = nullptr);

void L2Dev_Struct3d7p_Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream = nullptr);

class L2EqSolver : public L1EqSolver {
public:
    L2EqSolver(FLAG _type, INT _maxit, REAL _tol, REAL _relax_factor, FLAG _pc_type = SolverType::Empty, INT _pc_maxit = 5, REAL _pc_relax_factor = 1.0) : 
        L1EqSolver(_type, _maxit, _tol, _relax_factor, _pc_type, _pc_maxit, _pc_relax_factor) 
    {}

    void L2Dev_Struct3d7p_Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream = nullptr);
    void L2Dev_Struct3d7p_SOR(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream = nullptr);
    void L2Dev_Struct3d7p_PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream = nullptr);
    void L2Dev_Struct3d7p_Solve(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream = nullptr) {
        if (type == SolverType::Jacobi) {
            L2Dev_Struct3d7p_Jacobi(a, x, b, r, global, pdm, block_dim, cpm, stream);
        } else if (type == SolverType::SOR) {
            L2Dev_Struct3d7p_SOR(a, x, b, r, global, pdm, block_dim, cpm, stream);
        } else if (type == SolverType::PBiCGStab) {
            L2Dev_Struct3d7p_PBiCGStab(a, x, b, r, global, pdm, block_dim, cpm, stream);
        }
    }

protected:
    void L2Dev_Struct3d7p_JacobiPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream = nullptr);
    void L2Dev_Struct3d7p_SORPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream = nullptr);
    void L2Dev_Struct3d7p_Precondition(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream = nullptr) {
        if (pc_type == SolverType::Jacobi) {
            L2Dev_Struct3d7p_JacobiPC(a, x, b, pdm, block_dim, cpm, stream);
        } else if (pc_type == SolverType::SOR) {
            L2Dev_Struct3d7p_SORPC(a, x, b, pdm, block_dim, cpm, stream);
        }
    }

};

}

#endif
