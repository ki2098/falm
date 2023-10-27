#ifndef FALM_REGION_H
#define FALM_REGION_H

#include "util.h"

namespace Falm {

struct Region {
    INT3 shape;
    INT3 offset;
    INT   size;
    Region() : shape(INT3{0, 0, 0}), offset(INT3{0, 0, 0}), size(0) {}
    Region(INT3 _shape, INT3 _offset) : shape(_shape), offset(_offset), size(PRODUCT3(_shape)) {}
    Region(INT3 _shape, INT gc) : Region({_shape.x - gc*2, _shape.y - gc*2, _shape.z - gc*2}, {gc, gc, gc}) {}
    // Region(const Region &omap, INT guide) : Region(INTx3{omap.shape.x - guide * 2, omap.shape.y - guide * 2, omap.shape.z - guide * 2}, INTx3{guide, guide, guide}) {}

    Region transform(INT3 shape_trans, INT3 offset_trans) {
        INT3 new_shape = {
            shape.x + shape_trans.x,
            shape.y + shape_trans.y,
            shape.z + shape_trans.z
        };
        INT3 new_offset = {
            offset.x + offset_trans.x,
            offset.y + offset_trans.y,
            offset.z + offset_trans.z
        };
        return Region(new_shape, new_offset);
    }

};

}

#endif
