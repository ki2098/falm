#ifndef _DOM_H_
#define _DOM_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include "Util.cuh"

namespace FALM {

const unsigned int guide = 2;

struct DomCp {
    dim3 _size;
    dim3 _offset;
    unsigned int _num;

    DomCp(dim3 &size, dim3 &origin);
};

DomCp::DomCp(dim3 &size, dim3 &offset) : _size(size), _offset(offset) {
    _num = size.x * size.y * size.z;
}

struct Dom {
    DomCp  _h;
    DomCp *_d;
    Dom(dim3 &size, dim3 &offset);
    ~Dom();
};

Dom::Dom(dim3 &size, dim3 &offset) : _h(size, offset) {
    cudaMalloc(&_d, sizeof(DomCp));
    cudaMemcpy( _d, &_h, sizeof(DomCp), cudaMemcpyHostToDevice);
}

Dom::~Dom() {
    cudaFree(_d);
}

}

#endif