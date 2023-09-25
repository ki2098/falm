#ifndef FALM_STRUCTEQL2_H
#define FALM_STRUCTEQL2_H

#include "structEqL1.h"
#include "CPML2v2.h"

namespace Falm {

void devL2_Struct3d7p_MV(Matrix<double> &a, Matrix<double> &x, Matrix<double> &ax, Mapper &pdom, dim3 block_dim, CPMBase &cpm);

void devL2_Struct3d7p_Res(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &pdom, dim3 block_dim, CPMBase &cpm);

class L2EqSolver : public L1EqSolver {
public:
    L2EqSolver(unsigned int _type, int _maxit, double _tol, double _relax_factor, unsigned int _pc_type = SolverType::Empty, int _pc_maxit = 5, double _pc_relax_factor = 1.0) : 
        L1EqSolver(_type, _maxit, _tol, _relax_factor, _pc_type, _pc_maxit, _pc_relax_factor) 
    {}

    void devL2_Struct3d7p_Jacobi(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm);
    void devL2_Struct3d7p_SOR(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm);
    void devL2_Struct3d7p_PBiCGStab(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm);
    void devL2_Struct3d7p_Solve(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
        if (type == SolverType::Jacobi) {
            devL2_Struct3d7p_Jacobi(a, x, b, r, global, pdom, block_dim, cpm);
        } else if (type == SolverType::SOR) {
            devL2_Struct3d7p_SOR(a, x, b, r, global, pdom, block_dim, cpm);
        } else if (type == SolverType::PBiCGStab) {
            devL2_Struct3d7p_PBiCGStab(a, x, b, r, global, pdom, block_dim, cpm);
        }
    }

protected:
    void devL2_Struct3d7p_JacobiPC(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, dim3 block_dim, CPMBase &cpm);
    void devL2_Struct3d7p_SORPC(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, dim3 block_dim, CPMBase &cpm);
    void devL2_Struct3d7p_Precondition(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
        if (pc_type == SolverType::Jacobi) {
            devL2_Struct3d7p_JacobiPC(a, x, b, pdom, block_dim, cpm);
        } else if (pc_type == SolverType::SOR) {
            devL2_Struct3d7p_SORPC(a, x, b, pdom, block_dim, cpm);
        }
    }

private:
    void devL2_Struct3d7p_JacobiSweepBoundary(Matrix<double> &a, Matrix<double> &x, Matrix<double> &xp, Matrix<double> &b, Mapper &pdom, CPMBase &cpm, uint3 *boundary_shape, uint3 *boundary_offset) {
        if (cpm.neighbour[0] >= 0) {
            Mapper emap(boundary_shape[0], boundary_offset[0]);
            devL0_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, emap, dim3(1, 8, 8));
        }
        if (cpm.neighbour[1] >= 0) {
            Mapper wmap(boundary_shape[1], boundary_offset[1]);
            devL0_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, wmap, dim3(1, 8, 8));
        }
        if (cpm.neighbour[2] >= 0) {
            Mapper nmap(boundary_shape[2], boundary_offset[2]);
            devL0_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, nmap, dim3(8, 1, 8));
        }
        if (cpm.neighbour[3] >= 0) {
            Mapper smap(boundary_shape[3], boundary_offset[3]);
            devL0_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, smap, dim3(8, 1 ,8));
        }
        if (cpm.neighbour[4] >= 0) {
            Mapper tmap(boundary_shape[4], boundary_offset[4]);
            devL0_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, tmap, dim3(8, 8, 1));
        }
        if (cpm.neighbour[5] >= 0) {
            Mapper bmap(boundary_shape[5], boundary_offset[5]);
            devL0_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, bmap, dim3(8, 8, 1));
        }
    }

    void devL2_Struct3d7p_SORSweepBoundary(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, double omega, unsigned int color, Mapper &pdom, CPMBase &cpm, uint3 *boundary_shape, uint3 *boundary_offset) {
        if (cpm.neighbour[0] >= 0) {
            Mapper emap(boundary_shape[0], boundary_offset[0]);
            devL0_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, emap, dim3(1, 8, 8));
        }
        if (cpm.neighbour[1] >= 0) {
            Mapper wmap(boundary_shape[1], boundary_offset[1]);
            devL0_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, wmap, dim3(1, 8, 8));
        }
        if (cpm.neighbour[2] >= 0) {
            Mapper nmap(boundary_shape[2], boundary_offset[2]);
            devL0_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, nmap, dim3(8, 1, 8));
        }
        if (cpm.neighbour[3] >= 0) {
            Mapper smap(boundary_shape[3], boundary_offset[3]);
            devL0_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, smap, dim3(8, 1, 8));
        }
        if (cpm.neighbour[4] >= 0) {
            Mapper tmap(boundary_shape[4], boundary_offset[4]);
            devL0_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, tmap, dim3(8, 8, 1));
        }
        if (cpm.neighbour[5] >= 0) {
            Mapper bmap(boundary_shape[5], boundary_offset[5]);
            devL0_Struct3d7p_SORSweep(a, x, b, omega, color, pdom, bmap, dim3(8, 8, 1));
        }
    }
};

}

#endif
