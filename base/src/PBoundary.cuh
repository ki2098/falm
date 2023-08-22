#ifndef _PBOUNDARY_H_
#define _PBOUNDARY_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include "StructuredField.cuh"
#include "Util.cuh"
#include "Dom.cuh"

namespace FALM {

__global__ static void pb_east_kernel(FieldCp<double> &p, DomCp &dom) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &isz = dom._isz;
    dim3 &osz = dom._osz;
    unsigned int area = isz.y * isz.z;
    unsigned int g = dom._guide;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = osz.x - g;
        unsigned int oj = g + (idx / isz.z);
        unsigned int ok = g + (idx % isz.z);
        p(FALMUtil::d321(oi,oj,ok,osz)) = 0;
    }
}

__global__ static void pb_west_kernel(FieldCp<double> &p, DomCp &dom) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &isz = dom._isz;
    dim3 &osz = dom._osz;
    unsigned int area = isz.y * isz.z;
    unsigned int g = dom._guide;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = g - 1;
        unsigned int oj = g + (idx / isz.z);
        unsigned int ok = g + (idx % isz.z);
        p(FALMUtil::d321(oi,oj,ok,osz)) = p(FALMUtil::d321(oi+1,oj,ok,osz));
    }
}

__global__ static void pb_north_kernel(FieldCp<double> &p, DomCp &dom) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &isz = dom._isz;
    dim3 &osz = dom._osz;
    unsigned int area = isz.x * isz.z;
    unsigned int g = dom._guide;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = g + (idx / isz.z);
        unsigned int oj = osz.y - g;
        unsigned int ok = g + (idx % isz.z);
        p(FALMUtil::d321(oi,oj,ok,osz)) = p(FALMUtil::d321(oi,oj-1,ok,osz));
    }
}

__global__ static void pb_south_kernel(FieldCp<double> &p, DomCp &dom) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &isz = dom._isz;
    dim3 &osz = dom._osz;
    unsigned int area = isz.x * isz.z;
    unsigned int g = dom._guide;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = g + (idx / isz.z);
        unsigned int oj = g - 1;
        unsigned int ok = g + (idx % isz.z);
        p(FALMUtil::d321(oi,oj,ok,osz)) = p(FALMUtil::d321(oi,oj+1,ok,osz));
    }
}

__global__ static void pb_upper_kernel(FieldCp<double> &p, DomCp &dom) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &isz = dom._isz;
    dim3 &osz = dom._osz;
    unsigned int area = isz.x * isz.y;
    unsigned int g = dom._guide;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = g + (idx / isz.y);
        unsigned int oj = g + (idx % isz.y);
        unsigned int ok = osz.z - g;
        p(FALMUtil::d321(oi,oj,ok,osz)) = p(FALMUtil::d321(oi,oj,ok-1,osz));
    }
}

__global__ static void pb_lower_kernel(FieldCp<double> &p, DomCp &dom) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &isz = dom._isz;
    dim3 &osz = dom._osz;
    unsigned int area = isz.x * isz.y;
    unsigned int g = dom._guide;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = g + (idx / isz.y);
        unsigned int oj = g + (idx % isz.y);
        unsigned int ok = g - 1;
        p(FALMUtil::d321(oi,oj,ok,osz)) = p(FALMUtil::d321(oi,oj,ok+1,osz));
    }
}

static void pressure_boundary(Field<double> &p, Dom &dom, Dom &global, int mpi_size, int mpi_rank) {
    if (mpi_rank == 0) {
        pb_west_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d));
    }
    if (mpi_rank == mpi_size - 1) {
        pb_east_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d));
    }
    pb_north_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d));
    pb_south_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d));
    pb_upper_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d));
    pb_lower_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d));
}

}

#endif