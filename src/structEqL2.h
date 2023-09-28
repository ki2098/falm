#ifndef FALM_STRUCTEQL2_H
#define FALM_STRUCTEQL2_H

#include "structEqL1.h"
#include "CPM.h"

namespace Falm {

void L2Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Mapper &pdom, dim3 block_dim, CPMBase &cpm);

void L2Dev_Struct3d7p_Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &pdom, dim3 block_dim, CPMBase &cpm);

class L2EqSolver : public L1EqSolver {
public:
    L2EqSolver(FLAG _type, INT _maxit, REAL _tol, REAL _relax_factor, FLAG _pc_type = SolverType::Empty, INT _pc_maxit = 5, REAL _pc_relax_factor = 1.0) : 
        L1EqSolver(_type, _maxit, _tol, _relax_factor, _pc_type, _pc_maxit, _pc_relax_factor) 
    {}

    void L2Dev_Struct3d7p_Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm);
    void L2Dev_Struct3d7p_SOR(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm);
    void L2Dev_Struct3d7p_PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm);
    void L2Dev_Struct3d7p_Solve(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
        if (type == SolverType::Jacobi) {
            L2Dev_Struct3d7p_Jacobi(a, x, b, r, global, pdom, block_dim, cpm);
        } else if (type == SolverType::SOR) {
            L2Dev_Struct3d7p_SOR(a, x, b, r, global, pdom, block_dim, cpm);
        } else if (type == SolverType::PBiCGStab) {
            L2Dev_Struct3d7p_PBiCGStab(a, x, b, r, global, pdom, block_dim, cpm);
        }
    }

protected:
    void L2Dev_Struct3d7p_JacobiPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim, CPMBase &cpm);
    void L2Dev_Struct3d7p_SORPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim, CPMBase &cpm);
    void L2Dev_Struct3d7p_Precondition(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
        if (pc_type == SolverType::Jacobi) {
            L2Dev_Struct3d7p_JacobiPC(a, x, b, pdom, block_dim, cpm);
        } else if (pc_type == SolverType::SOR) {
            L2Dev_Struct3d7p_SORPC(a, x, b, pdom, block_dim, cpm);
        }
    }

private:
    void L2Dev_Struct3d7p_JacobiSweepBoundary(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &xp, Matrix<REAL> &b, Mapper &pdom, CPMBase &cpm, INTx3 *boundary_shape, INTx3 *boundary_offset) {
        if (cpm.neighbour[0] >= 0) {
            Mapper emap(boundary_shape[0], boundary_offset[0]);
            L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, emap, dim3(1, 8, 8));
        }
        if (cpm.neighbour[1] >= 0) {
            Mapper wmap(boundary_shape[1], boundary_offset[1]);
            L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, wmap, dim3(1, 8, 8));
        }
        if (cpm.neighbour[2] >= 0) {
            Mapper nmap(boundary_shape[2], boundary_offset[2]);
            L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, nmap, dim3(8, 1, 8));
        }
        if (cpm.neighbour[3] >= 0) {
            Mapper smap(boundary_shape[3], boundary_offset[3]);
            L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, smap, dim3(8, 1 ,8));
        }
        if (cpm.neighbour[4] >= 0) {
            Mapper tmap(boundary_shape[4], boundary_offset[4]);
            L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, tmap, dim3(8, 8, 1));
        }
        if (cpm.neighbour[5] >= 0) {
            Mapper bmap(boundary_shape[5], boundary_offset[5]);
            L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, bmap, dim3(8, 8, 1));
        }
    }

    void L2Dev_Struct3d7p_SORSweepBoundary(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, REAL omega, FLAG color, Mapper &pdom, CPMBase &cpm, INTx3 *boundary_shape, INTx3 *boundary_offset) {
        if (cpm.neighbour[0] >= 0) {
            Mapper emap(boundary_shape[0], boundary_offset[0]);
            L0Dev_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, emap, dim3(1, 8, 8));
        }
        if (cpm.neighbour[1] >= 0) {
            Mapper wmap(boundary_shape[1], boundary_offset[1]);
            L0Dev_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, wmap, dim3(1, 8, 8));
        }
        if (cpm.neighbour[2] >= 0) {
            Mapper nmap(boundary_shape[2], boundary_offset[2]);
            L0Dev_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, nmap, dim3(8, 1, 8));
        }
        if (cpm.neighbour[3] >= 0) {
            Mapper smap(boundary_shape[3], boundary_offset[3]);
            L0Dev_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, smap, dim3(8, 1, 8));
        }
        if (cpm.neighbour[4] >= 0) {
            Mapper tmap(boundary_shape[4], boundary_offset[4]);
            L0Dev_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, tmap, dim3(8, 8, 1));
        }
        if (cpm.neighbour[5] >= 0) {
            Mapper bmap(boundary_shape[5], boundary_offset[5]);
            L0Dev_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, bmap, dim3(8, 8, 1));
        }
    }
};

}

#endif