#ifndef FALM_MAPPER_H
#define FALM_MAPPER_H

#include "typedef.h"
#include "util.h"

namespace Falm {

struct Mapper {
    uint3 shape;
    uint3 offset;
    unsigned int size;
    Mapper() : shape(uint3{0, 0, 0}), offset(uint3{0, 0, 0}), size(0) {}
    Mapper(uint3 _shape, uint3 _offset) : shape(_shape), offset(_offset), size(PRODUCT3(_shape)) {}
};

}

#endif
