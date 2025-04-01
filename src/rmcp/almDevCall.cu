#include <math.h>
#include "almDevCall.h"
#include "../dev/devutil.cuh"
#include "../falmath.h"

#define BLADE_WIDTH 0.1
#define BLADE_THICK 0.1

namespace Falm {

namespace Rmcp {

__global__ void kernel_SetALMFlag(
    MatrixFrame<Int>  *vflag,
    MatrixFrame<Real> *vx,
    Real               tt,
    TurbineFrame      *vturbines,
    Int3               pdm_shape,
    Int3               map_shape,
    Int3               map_offset 
) {
    MatrixFrame<Int>  &flag = *vflag;
    MatrixFrame<Real> &x    = *vx;
    TurbineFrame &tf = *vturbines;

    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        Real3 point{{x(idx,0), x(idx,1), x(idx,2)}};

        bool something = false;
        for (size_t tid = 0; tid < tf.n_turbine; tid ++) {
            Real3 dxyz = point - tf.base[tid];
            dxyz = one_angle_frame_rotation(dxyz, tf.angle[tid], tf.angle_type[tid]);
            dxyz -= tf.hub[tid];

            Real dx = dxyz[0], dy = dxyz[1], dz = dxyz[2];
            Real rr = sqrt(dy * dy + dz * dz);
            Real th = atan2(dz, dy);
            if (th < 0) {
                th += 2 * Pi;
            }

            Real rt = tf.tip_rate[tid]*tt;
            const size_t Nb = tf.n_blade;
            for (size_t bid = 0; bid < Nb; bid ++) {
                Real rtblade = rt + (bid*2./Nb)*Pi;
                Real thblade = floormod(th - rtblade, 2.*Pi);
                bool isblade = (rr <= tf.radius && rr > tf.hub_radius);
                isblade = isblade && (thblade >= (Nb-1)*Pi / (Nb*2)) && (thblade <= (Nb+1)*Pi / (Nb*2));
                isblade = isblade && (rr <= BLADE_WIDTH / (2 * cos(thblade)) || rr <= BLADE_WIDTH / (2 * cos(thblade + Pi)));
                isblade = isblade && dx >= 0 && dx < BLADE_THICK;
                if (isblade) {
                    something = true;
                    flag(idx) = tid + 1;
                }
            }
        }
        if (!something) {
            flag(idx) = 0;
        }
    }
}


__global__ void kernel_SetALMFlag(
    const MatrixFrame<Int>  *vflag,
    const MatrixFrame<Real> *vx,
    Real               t,
    RmcpTurbine       *turbines,
    Int                nTurbine,
    Int3               pdm_shape,
    Int3               map_shape,
    Int3               map_offset
) {
    const MatrixFrame<Int>  &flag = *vflag;
    const MatrixFrame<Real> &x    = *vx;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        Real tx = x(idx, 0);
        Real ty = x(idx, 1);
        Real tz = x(idx, 2);

        bool something = false;
        for (Int _ti = 0; _ti < nTurbine; _ti ++) {
            RmcpTurbine &turb = turbines[_ti];
            Real3 dxyz = Real3{{tx, ty, tz}} - turb.pos;
            dxyz = turb.transform(dxyz);
            dxyz -= turb.rotpos;
            Real dx = dxyz[0], dy = dxyz[1], dz = dxyz[2];
            Real rr = sqrt(dy * dy + dz * dz);
            Real th = atan2(dz, dy);
            if (th < 0) {
                th += 2 * Pi;
            }
            Real rt = turb.tip * t;
            Real th1 = floormod(th - rt               , 2 * Pi);
            Real th2 = floormod(th - rt + 2 * Pi / 3.0, 2 * Pi);
            Real th3 = floormod(th - rt + 4 * Pi / 3.0, 2 * Pi);
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

void RmcpAlmDevCall::SetALMFlag(Matrix<Real> &x, Real t, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_SetALMFlag<<<grid_dim, block_dim, 0, 0>>>(alm_flag.devptr, x.devptr, t, wf.tdevptr, wf.nTurbine, pdm.shape, map.shape, map.offset);
}

void RmcpAlmDevCall::SetALMFlag(Matrix<Real> &x, Real t, const Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_SetALMFlag<<<grid_dim, block_dim>>>(alm_flag.devptr, x.devptr, t, turbines.devptr, pdm.shape, map.shape, map.offset);
}

__global__ void kernel_ALM(
    BHFrame             *vblades,
    MatrixFrame<Int>    *vflag,
    MatrixFrame<Real>   *vu,
    MatrixFrame<Real>   *vx,
    MatrixFrame<Real>   *vff,
    TurbineFrame        *vturbines,
    Int3                 pdm_shape,
    Int3                 map_shape,
    Int3                 map_offset
) {
    BHFrame           &blades   = *vblades;
    MatrixFrame<Int>  &flag     = *vflag;
    MatrixFrame<Real> &u        = *vu;
    MatrixFrame<Real> &x        = *vx;
    MatrixFrame<Real> &ff       = *vff;
    TurbineFrame      &turbines = *vturbines;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        Real3 point{{x(idx,0), x(idx,1), x(idx,2)}};

        Int  tflag = flag(idx);
        if (tflag > 0) {
            size_t tid = tflag - 1;
            Real3 xyz  = point - turbines.base[tid];
            Real3 uvw  = Real3{{u(idx,0), u(idx,1), u(idx,2)}} - turbines.base_velocity[tid];
            xyz = one_angle_frame_rotation(xyz, turbines.angle[tid], turbines.angle_type[tid]);
            uvw = one_angle_frame_rotation_dt(xyz, uvw, turbines.angle[tid], turbines.angular_velocity[tid], turbines.angle_type[tid]);
            Real3 dxyz = xyz - turbines.hub[tid];

            Real dx = dxyz[0], dy = dxyz[1], dz = dxyz[2];
            Real th = atan2(dz, dy);
            Real rr = sqrt(dy * dy + dz * dz);

            Real  ux  = uvw[0];
            Real  ut  = turbines.tip_rate[tid] * rr + uvw[1] * sin(th) - uvw[2] * cos(th);
            Real  urel2 = ux * ux + ut * ut;
            Real phi = atan(ux / ut);
            Real chord, twist, cl, cd;
            blades.get_airfoil_params(rr, rad2deg(phi), chord, twist, cl, cd);

            Real ps = (rr > BLADE_WIDTH)? asin(0.5 * BLADE_WIDTH / rr) : 0.5*Pi/turbines.n_blade;
            Real Cf = 0.5 * chord / (2 * rr * ps * BLADE_THICK);
            Real ffx = fabs((cl * cos(phi) + cd * sin(phi)) * Cf * urel2);
            Real fft = fabs((cl * sin(phi) - cd * cos(phi)) * Cf * urel2);
            Real3 ffxyz{{- ffx, fft * sin(th), - fft * cos(th)}};

            ffxyz = one_angle_frame_rotation(ffxyz, - turbines.angle[tid], turbines.angle_type[tid]);
            ff(idx, 0) = ffxyz[0];
            ff(idx, 1) = ffxyz[1];
            ff(idx, 2) = ffxyz[2];
            // printf("(%d %d %d) (%e %e %e) %e %e %e (%e %e %e) (%e %e %e)\n", i, j, k, ffxyz[0], ffxyz[1], ffxyz[2], ux, ut, rr, point[0], point[1], point[2], cl, cd, asin(0.5 * BLADE_WIDTH / rr));
        } else {
            ff(idx, 0) = 0;
            ff(idx, 1) = 0;
            ff(idx, 2) = 0;
        }
    }
}

__global__ void kernel_ALM(
    BHFrame             *vblades,
    MatrixFrame<Int>    *vflag,
    MatrixFrame<Real>   *vu,
    MatrixFrame<Real>   *vx,
    MatrixFrame<Real>   *vff,
    RmcpTurbine         *turbines,
    Int                  nTurbine,
    Int3                 pdm_shape,
    Int3                 map_shape,
    Int3                 map_offset
) {
    BHFrame           &blades = *vblades;
    MatrixFrame<Int>  &flag   = *vflag;
    MatrixFrame<Real> &u      = *vu;
    MatrixFrame<Real> &x      = *vx;
    MatrixFrame<Real> &ff     = *vff;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);

        Int  tflag = flag(idx);
        if (tflag > 0) {
            RmcpTurbine &turb = turbines[tflag - 1];
            Real3 dxyz = Real3{{x(idx, 0), x(idx, 1), x(idx, 2)}} - turb.pos;
            dxyz  = turb.transform(dxyz);
            dxyz -= turb.rotpos;
            Real dx = dxyz[0], dy = dxyz[1], dz = dxyz[2];
            Real th = atan2(dz, dy);
            Real rr = sqrt(dy * dy + dz * dz);
            Real3 uvw = turb.transform({{u(idx, 0), u(idx, 1), u(idx, 2)}});
            Real  ux  = uvw[0];
            Real  ut  = turb.tip * rr + uvw[1] * sin(th) - uvw[2] * cos(th);
            Real  uRel2 = ux * ux + ut * ut;
            Real phi = atan(ux / ut);
            Real chord, twist, cl, cd;
            blades.get_airfoil_params(rr, phi*180./Pi, chord, twist, cl, cd);
            Real Cf = 0.5 * chord / (2 * rr * asin(0.5 * turb.width / rr) * turb.thick);
            Real ffx = fabs((cl * cos(phi) + cd * sin(phi)) * Cf * uRel2);
            Real fft = fabs((cl * sin(phi) - cd * cos(phi)) * Cf * uRel2);
            Real3 ffxyz{{- ffx, fft * sin(th), - fft * cos(th)}};
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

__global__ void kernel_ALM(
    const MatrixFrame<Int>  *vflag,
    const MatrixFrame<Real> *vu,
    const MatrixFrame<Real> *vx,
    const MatrixFrame<Real> *vff,
    RmcpTurbine             *turbines,
    Int                      nTurbine,
    Int3                     pdm_shape,
    Int3                     map_shape,
    Int3                     map_offset
) {
    const MatrixFrame<Int> &flag = *vflag;
    const MatrixFrame<Real> &u   = *vu;
    const MatrixFrame<Real> &x   = *vx;
    const MatrixFrame<Real> &ff  = *vff;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        
        Int  tflag = flag(idx);
        if (tflag > 0) {
            RmcpTurbine &turb = turbines[tflag - 1];
            Real3 dxyz = Real3{{x(idx, 0), x(idx, 1), x(idx, 2)}} - turb.pos;
            dxyz  = turb.transform(dxyz);
            dxyz -= turb.rotpos;
            Real dx = dxyz[0], dy = dxyz[1], dz = dxyz[2];
            Real th = atan2(dz, dy);
            Real rr = sqrt(dy * dy + dz * dz);
            Real3 uvw = turb.transform({{u(idx, 0), u(idx, 1), u(idx, 2)}});
            Real  ux  = uvw[0];
            Real  ut  = turb.tip * rr + uvw[1] * sin(th) - uvw[2] * cos(th);
            Real  uRel2 = ux * ux + ut * ut;
            Real  chord = turb.chord(rr);
            Real  angle = turb.angle(rr);

            Real Cf = 0.5 * chord / (2 * rr * asin(0.5 * turb.width / rr) * turb.thick);
            Real phi = atan(ux / ut);
            Real alpha = phi * 180 / Pi - angle;
            Real Cd = turb.Cd(alpha);
            Real Cl = turb.Cl(alpha);

            Real ffx = fabs((Cl * cos(phi) + Cd * sin(phi)) * Cf * uRel2);
            Real fft = fabs((Cl * sin(phi) - Cd * cos(phi)) * Cf * uRel2);
            Real3 ffxyz{{- ffx, fft * sin(th), - fft * cos(th)}};
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

void RmcpAlmDevCall::ALM(BladeHandler &blades, Matrix<Real> &u, Matrix<Real> &x, Matrix<Real> &ff, Real t, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );

    kernel_ALM<<<grid_dim, block_dim>>>(blades.devptr, alm_flag.devptr, u.devptr, x.devptr, ff.devptr, wf.tdevptr, wf.nTurbine, pdm.shape, map.shape, map.offset);
}

void RmcpAlmDevCall::ALM(Matrix<Real> &u, Matrix<Real> &x, Matrix<Real> &ff, Real t, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );

    kernel_ALM<<<grid_dim, block_dim>>>(alm_flag.devptr, u.devptr, x.devptr, ff.devptr, wf.tdevptr, wf.nTurbine, pdm.shape, map.shape, map.offset);
}

void RmcpAlmDevCall::ALM(Matrix<Real> &u, Matrix<Real> &x, Matrix<Real> &ff, Real t, const Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_ALM<<<grid_dim, block_dim>>>(blades.devptr, alm_flag.devptr, u.devptr, x.devptr, ff.devptr, turbines.devptr, pdm.shape, map.shape, map.offset);
}

__global__ void kernel_CalcTorque(
    const MatrixFrame<Int>  *vflag,
    const MatrixFrame<Real> *vx,
    const MatrixFrame<Real> *vff,
    RmcpTurbine             *turbines,
    Int                      Tid,
    Real                    *partial_sum_dev,
    Int3                     pdm_shape,
    Int3                     map_shape,
    Int3                     map_offset
) {
    extern __shared__ Real cache[];
    const MatrixFrame<Int> &flag = *vflag;
    const MatrixFrame<Real> &x = *vx;
    const MatrixFrame<Real> &ff = *vff;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        Int tidx = IDX(threadIdx, blockDim);

        Real tmp  = 0;
        Int tflag = flag(idx);
        if (tflag == Tid + 1) {
            RmcpTurbine &turb = turbines[tflag - 1];
            Real3 dxyz = Real3{{x(idx, 0), x(idx, 1), x(idx, 2)}} - turb.pos;
            dxyz  = turb.transform(dxyz);
            dxyz -= turb.rotpos;
            Real dx = dxyz[0], dy = dxyz[1], dz = dxyz[2];
            Real rr = sqrt(dy * dy + dz * dz);
            Real3 fxyz = {{-ff(idx, 0), -ff(idx, 1), -ff(idx, 2)}};
            fxyz = turb.transform(fxyz);
            Real  fft  = sqrt(fxyz[1] * fxyz[1] + fxyz[2] * fxyz[2]);
            tmp = rr * fft;
        }
        cache[tidx] = tmp;
        __syncthreads();

        Int length = PRODUCT3(blockDim);
        while (length > 1) {
            Int cut = length / 2;
            Int reduce = length - cut;
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

void RmcpAlmDevCall::CalcTorque(Matrix<Real> &x, Matrix<Real> &ff, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    Int n_blocks = PRODUCT3(grid_dim);
    Int n_threads = PRODUCT3(block_dim);
    Real *partial_sum,*partial_sum_dev;
    falmErrCheckMacro(falmMalloc((void**)&partial_sum, sizeof(Real) * n_blocks));
    falmErrCheckMacro(falmMallocDevice((void**)&partial_sum_dev, sizeof(Real) * n_blocks));
    size_t shared_size = n_threads * sizeof(Real);

    for (Int __ti = 0; __ti < wf.nTurbine; __ti ++) {
        kernel_CalcTorque<<<grid_dim, block_dim, shared_size>>>(alm_flag.devptr, x.devptr, ff.devptr, wf.tdevptr, __ti, partial_sum_dev, pdm.shape, map.shape, map.offset);
        falmErrCheckMacro(falmMemcpy(partial_sum, partial_sum_dev, sizeof(Real) * n_blocks, MCP::Dev2Hst));
        Real sum = partial_sum[0];
        for (Int i = 0; i < n_blocks; i ++) {
            sum += partial_sum[i];
        }
        wf.tptr[__ti].torque = sum;
    }
    falmErrCheckMacro(falmFree(partial_sum));
    falmErrCheckMacro(falmFreeDevice(partial_sum_dev));
}

}

}
