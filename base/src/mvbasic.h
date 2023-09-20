#ifndef FALM_MATBASIC_H
#define FALM_MATBASIC_H

#include "matrix.h"
#include "mapper.h"

namespace Falm {

double dev_DotProduct(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim);

double dev_Norm2Sq(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 &block_dim);

double dev_MaxDiag(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 &block_dim);

void dev_ScaleMatrix(Matrix<double> &a, double scale, dim3 &block_dim);

}

#endif
