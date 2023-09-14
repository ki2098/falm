#ifndef FALM_MATBASIC_H
#define FALM_MATBASIC_H

#include "matrix.h"
#include "mapper.h"

namespace Falm {

double dev_calc_dot_product(Matrix<double> &a, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim);

double dev_calc_norm2_sq(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 &block_dim);

double dev_get_max_diag(Matrix<double> &a, Mapper &pdom, Mapper &map, dim3 &block_dim);

}

#endif