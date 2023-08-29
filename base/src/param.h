#ifndef FALM_PARAM_H
#define FALM_PARAM_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace FALM {

const double sor_omega = 1.2;

const dim3 block_size(8, 8, 2);

}

#endif