#ifndef FALM_MAPPER_H
#define FALM_MAPPER_H

#include "util.h"

namespace Falm {

struct Mapper {
    INTx3 shape;
    INTx3 offset;
    INT   size;
    Mapper() : shape(INTx3{0, 0, 0}), offset(INTx3{0, 0, 0}), size(0) {}
    Mapper(INTx3 _shape, INTx3 _offset) : shape(_shape), offset(_offset), size(PRODUCT3(_shape)) {}
    Mapper(Mapper &omap, INT guide) : Mapper(INTx3{omap.shape.x - guide * 2, omap.shape.y - guide * 2, omap.shape.z - guide * 2}, INTx3{guide, guide, guide}) {}

};

}

#endif
