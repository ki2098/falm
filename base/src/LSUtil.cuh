#ifndef _LS_UTIL_H_
#define _LS_UTIL_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "StructuredField.cuh"
#include "StructuredMesh.cuh"
#include "Util.cuh"

namespace FALM {

__global__ static void poisson_sor_kernel(FieldCp<double> &a, FieldCp<double> &x, FieldCp<double> &b, double omega, int color, dim3 &size, int start, int end) {
    int stride = FALMUtil::get_global_size();
    
}

}

#endif