#ifndef FALM_MATBASICDEV_H
#define FALM_MATBASICDEV_H

#include "matrix.h"
#include "mapper.h"

namespace Falm {

double devL1_DotProduct(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim);

double devL1_Norm2Sq(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 &block_dim);

double devL1_MaxDiag(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 &block_dim);

void devL1_ScaleMatrix(Matrix<double> &a, double scale, dim3 &block_dim);

}

#endif
