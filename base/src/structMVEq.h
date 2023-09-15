#ifndef FALM_STRUCTMVEQ_H
#define FALM_STRUCTMVEQ_H

#include "matrix.h"
#include "mapper.h"

namespace Falm {

void dev_Struct3d7p_MV(Matrix<double> &a, Matrix<double> &x, Matrix<double> &ax, Mapper &pdom, Mapper &map, dim3 &block_dim);

void dev_Struct3d7p_Res(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &pdom, Mapper &map, dim3 &block_dim);

class StructLEqSolver {
public:
    int           maxit;
    double          tol;
    int              it;
    double          err;
    double relax_factor;

    void dev_Struct3d7p_Jacobi(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim);
    void dev_Struct3d7p_SOR(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim);
    void dev_Struct3d7p_PBiCGStab(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim);
    
};

}

#endif