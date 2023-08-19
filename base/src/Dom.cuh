#ifndef _DOM_H_
#define _DOM_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include "Util.cuh"

namespace FALM {

struct DomCp {
    dim3           _isz;
    dim3           _osz;
    dim3        _origin;
    unsigned int _guide;
    unsigned int  _inum;
    unsigned int  _onum;

    DomCp(dim3 &size, dim3 &origin, unsigned int guide);
};

DomCp::DomCp(dim3 &size, dim3 &origin, unsigned int guide) : _isz(size), _osz(size.x+2*guide, size.y+2*guide, size.z+2*guide), _origin(origin), _guide(guide) {
    _inum = _isz.x * _isz.y * _isz.z;
    _onum = _osz.x * _osz.y * _osz.z;
}

struct Dom {
    DomCp  _h;
    DomCp *_d;
    Dom(dim3 &size, dim3 &origin, unsigned int guide);
    ~Dom();
};

Dom::Dom(dim3 &size, dim3 &origin, unsigned int guide) : _h(size, origin, guide) {
    cudaMalloc(&_d, sizeof(DomCp));
    cudaMemcpy( _d, &_h, sizeof(DomCp), cudaMemcpyHostToDevice);
}

Dom::~Dom() {
    cudaFree(_d);
}

}

#endif