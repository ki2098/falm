#ifndef _DOM_H_
#define _DOM_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include "Util.cuh"

namespace FALM {

const unsigned int guide = 2;

struct Dom {
    dim3 _size;
    dim3 _offset;
    unsigned int _num;

    Dom();
    Dom(const Dom &dom);
    Dom(dim3 size, dim3 origin);
    void set(dim3 size, dim3 origin);
};

Dom::Dom() : _size(0,0,0), _offset(0,0,0), _num(0) {}

Dom::Dom(dim3 size, dim3 offset) : _size(size), _offset(offset), _num(size.x*size.y*size.z) {}

void Dom::set(dim3 size, dim3 offset) {
    _size.x   =   size.x;
    _size.y   =   size.y;
    _size.z   =   size.z;
    _offset.x = offset.x;
    _offset.y = offset.y;
    _offset.z = offset.z;
    _num = size.x * size.y * size.z;
}

}

#endif