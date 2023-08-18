#ifndef _DOM_H_
#define _DOM_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include "Util.cuh"

struct Dom {
    dim3         _size;
    dim3       _origin;
    dim3    *_size_ptr;
    dim3  *_origin_ptr;
    unsigned int  _num;
    unsigned int    _g;
    unsigned int _idx0;
    unsigned int _idx1;

    Dom(dim3 &size, dim3 &origin, unsigned int gide);
    ~Dom();
};

Dom::Dom(dim3 &size, dim3 &origin, unsigned int gide) : _size(size), _origin(origin), _g(gide) {
    _num = size.x * size.y * size.z;
    _idx0 = FALMUtil::ijk2idx(_g, _g, _g, _size);
    _idx1 = FALMUtil::ijk2idx(_size.x - _g, _size.y - _g, _size.z - _g, _size);
    cudaMalloc(  &_size_ptr, sizeof(dim3));
    cudaMalloc(&_origin_ptr, sizeof(dim3));
    cudaMemcpy(  _size_ptr,   &_size, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(_origin_ptr, &_origin, sizeof(dim3), cudaMemcpyHostToDevice);
}

Dom::~Dom() {
    cudaFree(_size_ptr);
    cudaFree(_origin_ptr);
}

#endif