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
    Mapper(const Mapper &omap, INT guide) : Mapper(INTx3{omap.shape.x - guide * 2, omap.shape.y - guide * 2, omap.shape.z - guide * 2}, INTx3{guide, guide, guide}) {}

    Mapper transform(INTx3 shape_trans, INTx3 offset_trans) {
        INTx3 new_shape = {
            shape.x + shape_trans.x,
            shape.y + shape_trans.y,
            shape.z + shape_trans.z
        };
        INTx3 new_offset = {
            offset.x + offset_trans.x,
            offset.y + offset_trans.y,
            offset.z + offset_trans.z
        };
        return Mapper(new_shape, new_offset);
    }

};

}

#endif
