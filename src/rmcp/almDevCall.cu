#include <math.h>
#include "almDevCall.h"
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

        bool something = false;
        for (INT _ti = 0; _ti < nTurbine; _ti ++) {
            RmcpTurbine &turb = turbines[_ti];
            REAL3 dxyz = REAL3{{tx, ty, tz}} - turb.pos;
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
                something = true;
            }

            // printf("%d (%e %e %e) (%e %e %e) (%e %e %e)\n", _ti + 1, turb.pos[0], turb.pos[1], turb.pos[2], turb.rotpos[0], turb.rotpos[1], turb.rotpos[2], turb.roll, turb.pitch, turb.yaw);
        }
        if (!something) {
            flag(idx) = 0;
        }
    }
}

void RmcpAlmDevCall::SetALMFlag(Matrix<REAL> &x, REAL t, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim) {
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
            REAL3 dxyz = REAL3{{x(idx, 0), x(idx, 1), x(idx, 2)}} - turb.pos;
            dxyz  = turb.transform(dxyz);
            dxyz -= turb.rotpos;
            REAL dx = dxyz[0], dy = dxyz[1], dz = dxyz[2];
            REAL th = atan2(dz, dy);
            REAL rr = sqrt(dy * dy + dz * dz);
            REAL3 uvw = turb.transform({{u(idx, 0), u(idx, 1), u(idx, 2)}});
            REAL  ux  = uvw[0];
            REAL  ut  = turb.tip * rr + uvw[1] * sin(th) - uvw[2] * cos(th);
            REAL  uRel2 = ux * ux + ut * ut;
            REAL  chord = turb.chord(rr);
            REAL  angle = turb.angle(rr);

            REAL Cf = 0.5 * chord / (2 * rr * asin(0.5 * turb.width / rr) * turb.thick);
            REAL phi = atan(ux / ut);
            REAL alpha = phi * 180 / Pi - angle;
            REAL Cd = turb.Cd(alpha);
            REAL Cl = turb.Cl(alpha);

            REAL ffx = fabs((Cl * cos(phi) + Cd * sin(phi)) * Cf * uRel2);
            REAL fft = fabs((Cl * sin(phi) - Cd * cos(phi)) * Cf * uRel2);
            REAL3 ffxyz{{- ffx, fft * sin(th), - fft * cos(th)}};
            ffxyz = turb.invert_transform(ffxyz);
            ff(idx, 0) = ffxyz[0];
            ff(idx, 1) = ffxyz[1];
            ff(idx, 2) = ffxyz[2];
        } else {
            ff(idx, 0) = 0;
            ff(idx, 1) = 0;
            ff(idx, 2) = 0;
        }
    }
}

void RmcpAlmDevCall::ALM(Matrix<REAL> &u, Matrix<REAL> &x, Matrix<REAL> &ff, REAL t, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );

    kernel_ALM<<<grid_dim, block_dim>>>(alm_flag.devptr, u.devptr, x.devptr, ff.devptr, wf.tdevptr, wf.nTurbine, pdm.shape, map.shape, map.offset);
}

__global__ void kernel_CalcTorque(
    const MatrixFrame<INT>  *vflag,
    const MatrixFrame<REAL> *vx,
    const MatrixFrame<REAL> *vff,
    RmcpTurbine             *turbines,
    INT                      Tid,
    REAL                    *partial_sum_dev,
    INT3                     pdm_shape,
    INT3                     map_shape,
    INT3                     map_offset
) {
    extern __shared__ REAL cache[];
    const MatrixFrame<INT> &flag = *vflag;
    const MatrixFrame<REAL> &x = *vx;
    const MatrixFrame<REAL> &ff = *vff;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        INT tidx = IDX(threadIdx, blockDim);

        REAL tmp  = 0;
        INT tflag = flag(idx);
        if (tflag == Tid + 1) {
            RmcpTurbine &turb = turbines[tflag - 1];
            REAL3 dxyz = REAL3{{x(idx, 0), x(idx, 1), x(idx, 2)}} - turb.pos;
            dxyz  = turb.transform(dxyz);
            dxyz -= turb.rotpos;
            REAL dx = dxyz[0], dy = dxyz[1], dz = dxyz[2];
            REAL rr = sqrt(dy * dy + dz * dz);
            REAL3 fxyz = {{-ff(idx, 0), -ff(idx, 1), -ff(idx, 2)}};
            fxyz = turb.transform(fxyz);
            REAL  fft  = sqrt(fxyz[1] * fxyz[1] + fxyz[2] * fxyz[2]);
            tmp = rr * fft;
        }
        cache[tidx] = tmp;
        __syncthreads();

        INT length = PRODUCT3(blockDim);
        while (length > 1) {
            INT cut = length / 2;
            INT reduce = length - cut;
            if (tidx < cut) {
                cache[tidx] += cache[tidx + reduce];
            }
            __syncthreads();
            length = reduce;
        }
        if (tidx == 0) {
            partial_sum_dev[IDX(blockIdx, gridDim)] = cache[0];
        }
    }
}

void RmcpAlmDevCall::CalcTorque(Matrix<REAL> &x, Matrix<REAL> &ff, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    INT n_blocks = PRODUCT3(grid_dim);
    INT n_threads = PRODUCT3(block_dim);
    REAL *partial_sum = (REAL*)falmMalloc(sizeof(REAL) * n_blocks);
    REAL *partial_sum_dev = (REAL*)falmMallocDevice(sizeof(REAL) * n_blocks);
    size_t shared_size = n_threads * sizeof(REAL);

    for (INT __ti = 0; __ti < wf.nTurbine; __ti ++) {
        kernel_CalcTorque<<<grid_dim, block_dim, shared_size>>>(alm_flag.devptr, x.devptr, ff.devptr, wf.tdevptr, __ti, partial_sum_dev, pdm.shape, map.shape, map.offset);
        falmMemcpy(partial_sum, partial_sum_dev, sizeof(REAL) * n_blocks, MCpType::Dev2Hst);
        REAL sum = partial_sum[0];
        for (INT i = 0; i < n_blocks; i ++) {
            sum += partial_sum[i];
        }
        wf.tptr[__ti].torque = sum;
    }
    falmFree(partial_sum);
    falmFreeDevice(partial_sum_dev);
}

}
