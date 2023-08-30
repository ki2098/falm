#ifndef FALM_PARAM_H
#define FALM_PARAM_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace FALM {

const double sor_omega  =  1.2;
const int    ls_maxit   = 1000;
const double ls_epsilon = 1e-3;

const dim3 block_size(8, 8, 4);

}

#endif