#ifndef FALM_STRUCTMVEQ_H
#define FALM_STRUCTMVEQ_H

#include "matrix.h"
#include "mapper.h"

namespace Falm {

void dev_Struct3d7p_MV(Matrix<double> &a, Matrix<double> &x, Matrix<double> &ax, Mapper &pdom, Mapper &map, dim3 &block_dim);

void dev_Struct3d7p_Res(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &pdom, Mapper &map, dim3 &block_dim);

class StructLEqSolver {
public:
    int              maxit;
    double             tol;
    int                 it;
    double             err;
    double    relax_factor;
    int           pc_maxit;
    double pc_relax_factor;

    class SolverType {
    public:
        static const unsigned int Empty     = 0;
        static const unsigned int Jacobi    = 1;
        static const unsigned int SOR       = 2;
        static const unsigned int PBiCGStab = 4;
    };

    unsigned int    type;
    unsigned int pc_type;

    StructLEqSolver(unsigned int _type, int _maxit, double _tol, double _relax_factor, unsigned int _pc_type = SolverType::Empty, int _pc_maxit = 5, double _pc_relax_factor = 1.0) : 
        type(_type), maxit(_maxit), tol(_tol), relax_factor(_relax_factor),
        pc_type(_pc_type), pc_maxit(_pc_maxit), pc_relax_factor(_pc_relax_factor) 
    {}

    void dev_Struct3d7p_Jacobi(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim);
    void dev_Struct3d7p_SOR(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim);
    void dev_Struct3d7p_PBiCGStab(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim);
    void dev_Struct3d7p_Solve(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim) {
        if (type == SolverType::Jacobi) {
            dev_Struct3d7p_Jacobi(a, x, b, r, global, pdom, map, block_dim);
        } else if (type == SolverType::SOR) {
            dev_Struct3d7p_SOR(a, x, b, r, global, pdom, map, block_dim);
        } else if (type == SolverType::PBiCGStab) {
            dev_Struct3d7p_PBiCGStab(a, x, b, r, global, pdom, map, block_dim);
        }
    }

protected:
    void dev_Struct3d7p_JacobiSweep(Matrix<double> &a, Matrix<double> &x, Matrix<double> &xp, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim);
    void dev_Struct3d7p_SORSweep(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, double omega, unsigned int color, Mapper &pdom, Mapper &map, dim3 &block_dim);
    void dev_Struct3d7p_JAcobiPC(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim);
    void dev_Struct3d7p_SORPC(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim);
    void dev_Struct3d7p_Precondition(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim) {
        if (pc_type == SolverType::Jacobi) {
            dev_Struct3d7p_JAcobiPC(a, x, b, pdom, map, block_dim);
        } else if (pc_type == SolverType::SOR) {
            dev_Struct3d7p_SORPC(a, x, b, pdom, map, block_dim);
        }
    }
};

typedef StructLEqSolver::SolverType LSType;

}

#endif
