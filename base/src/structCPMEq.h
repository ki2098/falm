#ifndef FALM_STRUCTCPMEQ_H
#define FALM_STRUCTCPMEQ_H

#include "structMVEq.h"
#include "CPM.h"

namespace Falm {

void devL2_Struct3d7p_MV(Matrix<double> &a, Matrix<double> &x, Matrix<double> &ax, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm);

void devL2_Struct3d7p_Res(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm);

class L2EqSolver : public L1EqSolver {
public:
    L2EqSolver(unsigned int _type, int _maxit, double _tol, double _relax_factor, unsigned int _pc_type = SolverType::Empty, int _pc_maxit = 5, double _pc_relax_factor = 1.0) : 
        L1EqSolver(_type, _maxit, _tol, _relax_factor, _pc_type, _pc_maxit, _pc_relax_factor) 
    {}

    void devCPM_Struct3d7p_Jacobi(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm);
    void devCPM_Struct3d7p_SOR(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm);
    void devCPM_Struct3d7p_PBiCGStab(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm);
    void devCPM_Struct3d7p_Solve(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm) {
        if (type == SolverType::Jacobi) {
            devCPM_Struct3d7p_Jacobi(a, x, b, r, global, pdom, map, block_dim, cpm);
        } else if (type == SolverType::SOR) {
            devCPM_Struct3d7p_SOR(a, x, b, r, global, pdom, map, block_dim, cpm);
        } else if (type == SolverType::PBiCGStab) {
            devCPM_Struct3d7p_PBiCGStab(a, x, b, r, global, pdom, map, block_dim, cpm);
        }
    }

protected:
    void devCPM_Struct3d7p_JAcobiPC(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm);
    void devCPM_Struct3d7p_SORPC(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm);
    void devCPM_Struct3d7p_Precondition(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm) {
        if (pc_type == SolverType::Jacobi) {
            devCPM_Struct3d7p_JAcobiPC(a, x, b, pdom, map, block_dim, cpm);
        } else if (pc_type == SolverType::SOR) {
            devCPM_Struct3d7p_SORPC(a, x, b, pdom, map, block_dim, cpm);
        }
    }
};

}

#endif