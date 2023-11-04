#include <math.h>
#include "alm.h"
#include "../dev/devutil.cuh"
#include "../falmath.h"

namespace Falm {

__global__ void kernel_SetALMFlag(
    const MatrixFrame<INT>  *vflag,
    const MatrixFrame<REAL> *vx,
    REAL               t,
    RmcpTurbine       *turbines,
    INT                nTurbine,
    INT3               pdm_shape,
    INT3               map_shape,
    INT3               map_offset
) {
    const MatrixFrame<INT>  &flag = *vflag;
    const MatrixFrame<REAL> &x    = *vx;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdm_shape);
        REAL tx = x(idx, 0);
        REAL ty = x(idx, 1);
        REAL tz = x(idx, 2);

        for (INT _ti = 0; _ti < nTurbine; _ti ++) {
            RmcpTurbine &tur = turbines[_ti];
            REAL3 dxyz = REAL3{tx, ty, tz} - tur.pos;
            dxyz = tur.transform(dxyz);
        }
    }
}

}