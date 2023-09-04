#ifndef FALM_CPM_H
#define FALM_CPM_H

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#include "Mapper.cuh"
#include "Util.cuh"
#include "param.h"

namespace FALM {

namespace CPM {

const unsigned int EMPTY = 0;
const unsigned int IN    = 1;
const unsigned int OUT   = 2;
const unsigned int INOUT = IN | OUT;

struct Buffer {
    void           *ptr;
    unsigned int   size;
    size_t        dtype;
    unsigned int    loc;
    unsigned int iotype;
    Mapper          map;
    Buffer() : ptr(nullptr), size(0), loc(LOC::NONE), iotype(EMPTY) {}
    void release() {
        if (loc == LOC::HOST) {
            free(ptr);
        } else if (loc == LOC::DEVICE) {
            cudaFree(ptr);
        }
        ptr = nullptr;
        size = 0;
        loc = LOC::NONE;
        iotype = EMPTY;
    }

    void alloc(dim3 &vsize, dim3 &voffset, size_t vdtype, unsigned int vloc, unsigned int viotype) {
        assert(ptr == nullptr && size == 0 && loc == LOC::NONE && iotype == EMPTY);
        map.set(vsize, voffset);
        size = map.num;
        dtype = vdtype;
        loc = vloc;
        iotype = viotype;
        if (loc == LOC::HOST) {
            ptr = malloc(dtype * size);
        } else if (loc == LOC::DEVICE) {
            cudaMalloc(&ptr, dtype * size);
        }
    }

    void alloc(Mapper &domain, unsigned int color, dim3 &vsize, dim3 &voffset, size_t vdtype, unsigned int vloc, unsigned int viotype) {
        assert(ptr == nullptr && size == 0 && loc == LOC::NONE && iotype == EMPTY);
        map.set(vsize, voffset);
        unsigned int ref_color = (UTIL::dim3_sum(domain.offset) + UTIL::dim3_sum(map.offset)) % 2;
        size = map.num / 2;
        if (ref_color == color && map.num % 2 == 1) {
            size ++;
        }
        dtype = vdtype;
        loc = vloc;
        iotype = viotype;
        if (loc == LOC::HOST) {
            ptr = malloc(dtype * size);
        } else if (loc == LOC::DEVICE) {
            cudaMalloc(&ptr, dtype * size);
        }
    }
};

}

}

#endif