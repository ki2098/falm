#ifndef _PBOUNDARY_H_
#define _PBOUNDARY_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include "StructuredField.cuh"
#include "Util.cuh"
#include "Dom.cuh"

namespace FALM {

__global__ static void pb_east_kernel(FieldCp<double> &p, DomCp &dom, DomCp &inner) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &osz =   dom._size;
    dim3 &isz = inner._size;
    dim3 &ift = inner._offset;
    unsigned int area = isz.y * isz.z;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = ift.x + isz.x;
        unsigned int oj = ift.y + (idx / isz.z);
        unsigned int ok = ift.z + (idx % isz.z);
        p(FALMUtil::d321(oi,oj,ok,osz)) = 0;
    }
}

__global__ static void pb_west_kernel(FieldCp<double> &p, DomCp &dom, DomCp &inner) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &osz =   dom._size;
    dim3 &isz = inner._size;
    dim3 &ift = inner._offset;
    unsigned int area = isz.y * isz.z;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = ift.x - 1;
        unsigned int oj = ift.y + (idx / isz.z);
        unsigned int ok = ift.z + (idx % isz.z);
        p(FALMUtil::d321(oi,oj,ok,osz)) = 0;
    }
}

__global__ static void pb_north_kernel(FieldCp<double> &p, DomCp &dom, DomCp &inner) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &osz =   dom._size;
    dim3 &isz = inner._size;
    dim3 &ift = inner._offset;
    unsigned int area = isz.x * isz.z;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = ift.x + (idx / isz.z);
        unsigned int oj = ift.y + isz.y;
        unsigned int ok = ift.z + (idx % isz.z);
        p(FALMUtil::d321(oi,oj,ok,osz)) = p(FALMUtil::d321(oi,oj-1,ok,osz));
    }
}

__global__ static void pb_south_kernel(FieldCp<double> &p, DomCp &dom, DomCp &inner) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &osz =   dom._size;
    dim3 &isz = inner._size;
    dim3 &ift = inner._offset;
    unsigned int area = isz.x * isz.z;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = ift.x + (idx / isz.z);
        unsigned int oj = ift.y - 1;
        unsigned int ok = ift.z + (idx % isz.z);
        p(FALMUtil::d321(oi,oj,ok,osz)) = p(FALMUtil::d321(oi,oj-1,ok,osz));
    }
}

__global__ static void pb_upper_kernel(FieldCp<double> &p, DomCp &dom, DomCp &inner) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &osz =   dom._size;
    dim3 &isz = inner._size;
    dim3 &ift = inner._offset;
    unsigned int area = isz.x * isz.y;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = ift.x + (idx / isz.y);
        unsigned int oj = ift.y + (idx % isz.y);
        unsigned int ok = ift.z + isz.z;
        p(FALMUtil::d321(oi,oj,ok,osz)) = p(FALMUtil::d321(oi,oj,ok-1,osz));
    }
}

__global__ static void pb_lower_kernel(FieldCp<double> &p, DomCp &dom, DomCp &inner) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &osz =   dom._size;
    dim3 &isz = inner._size;
    dim3 &ift = inner._offset;
    unsigned int area = isz.x * isz.y;
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < area; idx += stride) {
        unsigned int oi = ift.x + (idx / isz.y);
        unsigned int oj = ift.y + (idx % isz.y);
        unsigned int ok = ift.z - 1;
        p(FALMUtil::d321(oi,oj,ok,osz)) = p(FALMUtil::d321(oi,oj,ok-1,osz));
    }
}

static void pressure_boundary(Field<double> &p, Dom &dom, Dom &global, Dom &inner, int mpi_size, int mpi_rank) {
    if (mpi_rank == 0) {
        pb_west_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d), *(inner._d));
    }
    if (mpi_rank == mpi_size - 1) {
        pb_east_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d), *(inner._d));
    }
    pb_north_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d), *(inner._d));
    pb_south_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d), *(inner._d));
    pb_upper_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d), *(inner._d));
    pb_lower_kernel<<<n_blocks, n_threads>>>(*(p._dd), *(dom._d), *(inner._d));
}

}

#endif