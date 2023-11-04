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
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        REAL tx = x(idx, 0);
        REAL ty = x(idx, 1);
        REAL tz = x(idx, 2);

        for (INT _ti = 0; _ti < nTurbine; _ti ++) {
            RmcpTurbine &turb = turbines[_ti];
            REAL3 dxyz = REAL3{tx, ty, tz} - turb.pos;
            dxyz = turb.transform(dxyz);
            dxyz = dxyz - turb.rotpos;
            REAL dx = dxyz[0], dy = dxyz[1], dz = dxyz[2];
            REAL rr = sqrt(dy * dy + dz * dz);
            REAL th = atan2(dz, dy);
            if (th < 0) {
                th += 2 * Pi;
            }
            REAL rt = turb.tip * t;
            REAL th1 = floormod(th - rt               , 2 * Pi);
            REAL th2 = floormod(th - rt + 2 * Pi / 3.0, 2 * Pi);
            REAL th3 = floormod(th - rt + 4 * Pi / 3.0, 2 * Pi);
            bool bld1 = (rr <= turb.R && rr > turb.hub) && (th1 >= Pi / 3.0) && (th1 <= 2.0 * Pi / 3.0) && (rr <= turb.width / (2 * cos(th1)) || rr <= turb.width / (2 * cos(th1 + Pi)));
            bool bld2 = (rr <= turb.R && rr > turb.hub) && (th2 >= Pi / 3.0) && (th2 <= 2.0 * Pi / 3.0) && (rr <= turb.width / (2 * cos(th2)) || rr <= turb.width / (2 * cos(th2 + Pi)));
            bool bld3 = (rr <= turb.R && rr > turb.hub) && (th3 >= Pi / 3.0) && (th3 <= 2.0 * Pi / 3.0) && (rr <= turb.width / (2 * cos(th3)) || rr <= turb.width / (2 * cos(th3 + Pi)));

            if ((bld1 || bld2 || bld3) && dx >= 0 && dx < turb.thick) {
                flag(idx) = _ti + 1;
            } else {
                flag(idx) = 0;
            }
        }
    }
}

void RmcpAlm::Rmcp_SetALMFlag(Matrix<REAL> &x, REAL t, RmcpWindfarm &wf, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region  map(pdm.shape, cpm.gc);
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_SetALMFlag<<<grid_dim, block_dim, 0, 0>>>(alm_flag.devptr, x.devptr, t, wf.tdevptr, wf.nTurbine, pdm.shape, map.shape, map.offset);
}

__global__ void kernel_ALM(
    const MatrixFrame<INT>  *vflag,
    const MatrixFrame<REAL> *vu,
    const MatrixFrame<REAL> *vx,
    const MatrixFrame<REAL> *vff,
    RmcpTurbine             *turbines,
    INT                      nTurbine,
    INT3                     pdm_shape,
    INT3                     map_shape,
    INT3                     map_offset
) {
    const MatrixFrame<INT> &flag = *vflag;
    const MatrixFrame<REAL> &u   = *vu;
    const MatrixFrame<REAL> &x   = *vx;
    const MatrixFrame<REAL> &ff  = *vff;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        
        INT  tflag = flag(idx);
        if (tflag > 0) {
            RmcpTurbine &turb = turbines[tflag - 1];
            REAL3 dxyz = REAL3{x(idx, 0), x(idx, 1), x(idx, 2)} - turb.pos;
            dxyz  = turb.transform(dxyz);
            dxyz -= turb.rotpos;
            REAL dx = dxyz[0], dy = dxyz[1], dz = dxyz[2];

            
        }
    }
}

}