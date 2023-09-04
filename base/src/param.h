#ifndef FALM_PARAM_H
#define FALM_PARAM_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace FALM {

const double sor_omega  =  1.2;
const int    ls_maxit   = 1000;
const double ls_epsilon = 1e-3;

const unsigned int block_dim_x = 8;
const unsigned int block_dim_y = 8;
const unsigned int block_dim_z = 4;

}

#endif