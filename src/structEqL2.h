#ifndef FALM_STRUCTEQL2_H
#define FALM_STRUCTEQL2_H

#include "structEqL1.h"
#include "CPM.h"

extern Falm::STREAM boundaryStream[6];

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
        
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 block_dim(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                Mapper map(boundary_shape[fid], boundary_offset[fid]);
                // falmCreateStream(&boundaryStream[fid]);
                L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, map, block_dim, boundaryStream[fid]);
            }
        }
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                falmStreamSync(boundaryStream[fid]);
                // falmDestroyStream(boundaryStream[fid]);
            }
        }
    }

    void L2Dev_Struct3d7p_SORSweepBoundary(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, REAL omega, FLAG color, Mapper &pdom, CPMBase &cpm, INTx3 *boundary_shape, INTx3 *boundary_offset) {
        
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 block_dim(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                Mapper map(boundary_shape[fid], boundary_offset[fid]);
                // falmCreateStream(&boundaryStream[fid]);
                L0Dev_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, map, block_dim, boundaryStream[fid]);
            }
        }
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                falmStreamSync(boundaryStream[fid]);
                // falmDestroyStream(boundaryStream[fid]);
            }
        }
    }
};

}

#endif
