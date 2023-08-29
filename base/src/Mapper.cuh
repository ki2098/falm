#ifndef FALM_MAPPER_CUH
#define FALM_MAPPER_CUH

namespace FALM {

const int guide = 2;

struct Mapper {
    dim3 size;
    dim3 offset;
    unsigned int num;
    Mapper() : size(0,0,0), offset(0,0,0), num(0) {}
    Mapper(dim3 vsize, dim3 voffset) : size(vsize), offset(voffset) {num = size.x * size.y * size.z;}
    void set(dim3 vsize, dim3 voffset);
};

void Mapper::set(dim3 vsize, dim3 voffset) {
    size.x   =   vsize.x;
    size.y   =   vsize.y;
    size.z   =   vsize.z;
    offset.x = voffset.x;
    offset.y = voffset.y;
    offset.z = voffset.z;
    num = size.x * size.y * size.z;
}

}

#endif