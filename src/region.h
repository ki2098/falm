#ifndef FALM_REGION_H
#define FALM_REGION_H

#include "util.h"

namespace Falm {

struct Region {
    Int3 shape;
    Int3 offset;
    Int   size;
    Region() : shape(Int3{{0, 0, 0}}), offset(Int3{{0, 0, 0}}), size(0) {}
    Region(Int3 _shape, Int3 _offset) : shape(_shape), offset(_offset), size(PRODUCT3(_shape)) {}
    Region(Int3 _shape, Int gc) : Region({{_shape[0] - gc*2, _shape[1] - gc*2, _shape[2] - gc*2}}, {{gc, gc, gc}}) {}
    // Region(const Region &omap, INT guide) : Region(INTx3{omap.shape[0] - guide * 2, omap.shape[1] - guide * 2, omap.shape[2] - guide * 2}, INTx3{guide, guide, guide}) {}

    Region transform(Int3 shape_trans, Int3 offset_trans) {
        Int3 new_shape = {{
            shape[0] + shape_trans[0],
            shape[1] + shape_trans[1],
            shape[2] + shape_trans[2]
        }};
        Int3 new_offset = {{
            offset[0] + offset_trans[0],
            offset[1] + offset_trans[1],
            offset[2] + offset_trans[2]
        }};
        return Region(new_shape, new_offset);
    }

};

}

#endif
