#ifndef FALM_STRUCTEQL1_H
#define FALM_STRUCTEQL1_H

#include "matrix.h"
#include "mapper.h"

namespace Falm {

void L0Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Mapper &pdom, Mapper &map, dim3 block_dim);

void L0Dev_Struct3d7p_Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &pdom, Mapper &map, dim3 block_dim);

static inline void L1Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    L0Dev_Struct3d7p_MV(a, x, ax, pdom, map, block_dim);
}

static inline void L1Dev_Struct3d7p_Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    L0Dev_Struct3d7p_Res(a, x, b, r, pdom, map, block_dim);
}

class L1EqSolver {
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

    L1EqSolver(FLAG _type, INT _maxit, REAL _tol, REAL _relax_factor, FLAG _pc_type = SolverType::Empty, INT _pc_maxit = 5, REAL _pc_relax_factor = 1.0) : 
        type(_type), maxit(_maxit), tol(_tol), relax_factor(_relax_factor),
        pc_type(_pc_type), pc_maxit(_pc_maxit), pc_relax_factor(_pc_relax_factor) 
    {}

    void L1Dev_Struct3d7p_Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim);
    void L1Dev_Struct3d7p_SOR(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim);
    void L1Dev_Struct3d7p_PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim);
    void L1Dev_Struct3d7p_Solve(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim) {
        if (type == SolverType::Jacobi) {
            L1Dev_Struct3d7p_Jacobi(a, x, b, r, global, pdom, block_dim);
        } else if (type == SolverType::SOR) {
            L1Dev_Struct3d7p_SOR(a, x, b, r, global, pdom, block_dim);
        } else if (type == SolverType::PBiCGStab) {
            L1Dev_Struct3d7p_PBiCGStab(a, x, b, r, global, pdom, block_dim);
        }
    }

protected:
    void L0Dev_PBiCGStab1(Matrix<REAL> &p, Matrix<REAL> &q, Matrix<REAL> &r, REAL beta, REAL omega, Mapper &pdom, Mapper &map, dim3 block_dim);
    void L0Dev_PBiCGStab2(Matrix<REAL> &s, Matrix<REAL> &q, Matrix<REAL> &r, REAL alpha, Mapper &pdom, Mapper &map, dim3 block_dim);
    void L0Dev_PBiCGStab3(Matrix<REAL> &x, Matrix<REAL> &pp, Matrix<REAL> &ss, REAL alpha, REAL omega, Mapper &pdom, Mapper &map, dim3 block_dim);
    void L0Dev_PBiCGStab4(Matrix<REAL> &r, Matrix<REAL> &s, Matrix<REAL> &t, REAL omega, Mapper &pdom, Mapper &map, dim3 block_dim);
    void L0Dev_Struct3d7p_JacobiSweep(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &xp, Matrix<REAL> &b, Mapper &pdom, Mapper &map, dim3 block_dim);
    void L0Dev_Struct3d7p_SORSweep(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, REAL omega, INT color, Mapper &pdom, Mapper &map, dim3 block_dim);
    void L1Dev_Struct3d7p_JacobiPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim);
    void L1Dev_Struct3d7p_SORPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim);
    void L1Dev_Struct3d7p_Precondition(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim) {
        if (pc_type == SolverType::Jacobi) {
            L1Dev_Struct3d7p_JacobiPC(a, x, b, pdom, block_dim);
        } else if (pc_type == SolverType::SOR) {
            L1Dev_Struct3d7p_SORPC(a, x, b, pdom, block_dim);
        }
    }



};

typedef L1EqSolver::SolverType LSType;

}

#endif
