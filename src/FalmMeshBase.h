#ifndef FALM_FALMSCMESH_H
#define FALM_FALMSCMESH_H

#include "matrix.h"

namespace Falm {

struct FalmBaseMesh {
    Matrix<Real> x, y, z, hx, hy, hz;

    void alloc(Int imax, Int jmax, Int kmax, Flag hdctype) {
        x.alloc(imax, 1, hdctype);
        y.alloc(jmax, 1, hdctype);
        z.alloc(kmax, 1, hdctype);
        hx.alloc(imax, 1, hdctype);
        hy.alloc(jmax, 1, hdctype);
        hz.alloc(kmax, 1, hdctype);
    }

    void sync(Flag mcptype) {
        x.sync(mcptype);
        y.sync(mcptype);
        z.sync(mcptype);
        hx.sync(mcptype);
        hy.sync(mcptype);
        hz.sync(mcptype);
    }

    void release() {
        x.release();
        y.release();
        z.release();
        hx.release();
        hy.release();
        hz.release();
    }
};

}

#endif