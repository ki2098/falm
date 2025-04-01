#include <math.h>
#include "almDevCall.h"
#include "../dev/devutil.cuh"
#include "../falmath.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace Falm {

namespace Alm {

template <typename T>
__host__ __device__ static Int find_index(T *seq, T value, Int size) {
    if (value < seq[0]) {
        return -1;
    } else if (value >= seq[size-1]) {
        return size-1;
    } else {
        for (Int i = 0; i < size-1; i ++) {
            if (seq[i] <= value && value < seq[i+1]) {
                return i;
            }
        }
        return size-1;
    }
}

__host__ __device__ inline Real trilinear_interpolation(
    Real xd, Real yd, Real zd,
    Real c0, Real c1, Real c2, Real c3, Real c4, Real c5, Real c6, Real c7
) {
    Real c01 = c0*(1. - xd) + c1*xd;
    Real c23 = c2*(1. - xd) + c3*xd;
    Real c45 = c4*(1. - xd) + c5*xd;
    Real c67 = c6*(1. - xd) + c7*xd;

    Real c0123 = c01*(1. - yd) + c23*yd;
    Real c4567 = c45*(1. - yd) + c67*yd;

    return c0123*(1. - zd) + c4567*zd;
}

__global__ void kernel_UpdateTurbineAngles(
    TurbineFrame *vturbines,
    Real t
) {
    TurbineFrame &turbines = *vturbines;
    size_t thread_id = GLOBAL_THREAD_IDX();
    if (thread_id < turbines.n_turbine) {
        auto angle_type = int(turbines.angle_type[thread_id]);
        auto motion = turbines.motion[thread_id];
        Real3 angle{{0,0,0}}, angular_velocity{{0,0,0}};
        if (angle_type) {
            angle[angle_type - 1] = motion[0]*sin(motion[1]*t + motion[2]);
            angular_velocity[angle_type - 1] = motion[0]*motion[1]*cos(motion[1]*t + motion[2]);
        }
        turbines.angle[thread_id] = angle;
        turbines.angular_velocity[thread_id] = angular_velocity;
        // printf("turbine %d (%lf %lf %lf) (%lf %lf %lf)\n", thread_id, turbines.angle[thread_id][0], turbines.angle[thread_id][1], turbines.angle[thread_id][2], turbines.angular_velocity[thread_id][0], turbines.angular_velocity[thread_id][1], turbines.angular_velocity[thread_id][2]);
    }
}

__global__ void kernel_UpdateAPX(
    MatrixFrame<Real> *vx,
    MatrixFrame<Real> *vy,
    MatrixFrame<Real> *vz,
    // MatrixFrame<INT> *vxoffset,
    // MatrixFrame<INT> *vyoffset,
    // MatrixFrame<INT> *vzoffset,
    TurbineFrame *vturbines,
    APFrame *vaps,
    Int n_ap_per_blade,
    Real t,
    Int gc,
    int rank
) {
    const MatrixFrame<Real> &x = *vx;
    const MatrixFrame<Real> &y = *vy;
    const MatrixFrame<Real> &z = *vz;
    // const MatrixFrame<INT> &xoffset = *vxoffset;
    // const MatrixFrame<INT> &yoffset = *vyoffset;
    // const MatrixFrame<INT> &zoffset = *vzoffset;
    const TurbineFrame &turbines = *vturbines;
    APFrame &aps = *vaps;

    size_t thread_id = GLOBAL_THREAD_IDX();
    if (thread_id < aps.apcount) {
        const Int ap_id = thread_id;
        const Int n_turbine = turbines.n_turbine;
        const Int n_blade = turbines.n_blade;
        const Int n_ap_per_turbine = n_ap_per_blade*n_blade;
        const Int turbine_id = ap_id/n_ap_per_turbine;
        const Int blade_id = (ap_id%n_ap_per_turbine)/n_ap_per_blade;
        const Real tip = turbines.tip_rate[turbine_id];

        double theta0 = (2*Pi/n_blade)*blade_id;
        // t = floormod(t, 2*Pi/tip);
        double theta  = tip*t + theta0;

        const Real3 &hub = turbines.hub[turbine_id];
        const Real3 &base = turbines.base[turbine_id];
        const Real3 &angle = turbines.angle[turbine_id];
        const EulerAngle angle_type = turbines.angle_type[turbine_id];
        const Real apr = aps.r[ap_id];

        Real3 coordinate1 = hub;
        coordinate1[1] += apr*cos(theta);
        coordinate1[2] += apr*sin(theta);
        Real3 coordinate0 = one_angle_frame_rotation(coordinate1, - angle, angle_type) + base;

        aps.xyz[ap_id] = coordinate0;
        Int apI = find_index(x.ptr, coordinate0[0], x.size);
        Int apJ = find_index(y.ptr, coordinate0[1], y.size);
        Int apK = find_index(z.ptr, coordinate0[2], z.size);
        aps.ijk[ap_id] = Int3{{apI, apJ, apK}};

        if (apI >= gc-1 && apI < x.size-gc-1 && apJ >= gc-1 && apJ < y.size-gc-1 && apK >= gc-1 && apK < z.size-gc-1) {
            aps.rank[ap_id] = rank;
        } else {
            aps.rank[ap_id] = -1;
        }

        // INT apRankI = find_index(xoffset.ptr, apI, xoffset.size);
        // INT apRankJ = find_index(yoffset.ptr, apJ, yoffset.size);
        // INT apRankK = find_index(zoffset.ptr, apK, zoffset.size);
        // aps.rank[ap_id] = IDX(apRankI, apRankJ, apRankK, mpi_shape);
        // printf("%d %d: %d (%lf %lf %lf) (%d %d %d)\n", rank, ap_id, aps.rank[ap_id], coordinate0[0], coordinate0[1], coordinate0[2], apI, apJ, apK);
    }
}

__global__ void kernel_CalcAPForce(
    MatrixFrame<Real> *vx,
    MatrixFrame<Real> *vy,
    MatrixFrame<Real> *vz,
    MatrixFrame<Real> *vuvw,
    TurbineFrame *vturbines,
    APFrame *vaps,
    Int n_ap_per_blade,
    Int3 shape,
    Real t,
    int rank
) {
    const MatrixFrame<Real> &x = *vx;
    const MatrixFrame<Real> &y = *vy;
    const MatrixFrame<Real> &z = *vz;
    const MatrixFrame<Real> &uvw = *vuvw;
    const TurbineFrame &turbines = *vturbines;
    APFrame &aps = *vaps;

    size_t thread_id = GLOBAL_THREAD_IDX();
    if (thread_id < aps.apcount) {
        const Int ap_id = thread_id;
        const Int n_turbine = turbines.n_turbine;
        const Int n_blade = turbines.n_blade;
        const Int n_ap_per_turbine = n_ap_per_blade*n_blade;
        const Int turbine_id = ap_id/n_ap_per_turbine;
        const Int blade_id = (ap_id%n_ap_per_turbine)/n_ap_per_blade;
        const Real tip = turbines.tip_rate[turbine_id];
        const Real3 &hub = turbines.hub[turbine_id];
        const Real3 &angle = turbines.angle[turbine_id];
        const Real3 &angular_velocity = turbines.angular_velocity[turbine_id];
        const EulerAngle angle_type = turbines.angle_type[turbine_id];
        const Real apr = aps.r[ap_id];
        const Real dr_per_ap = (turbines.radius - turbines.hub_radius)/n_ap_per_blade;
        if (aps.rank[ap_id] == rank) {
            // printf("%lf\n", dr_per_ap);
            Int3 apijk = aps.ijk[ap_id];
            Int  i0 = apijk[0], i1 = apijk[0] + 1;
            Int  j0 = apijk[1], j1 = apijk[1] + 1;
            Int  k0 = apijk[2], k1 = apijk[2] + 1;
            Real x0 = x(i0), x1 = x(i1);
            Real y0 = y(j0), y1 = y(j1);
            Real z0 = z(k0), z1 = z(k1);
            Real3 apxyz = aps.xyz[ap_id];
            Real xd = fabs(apxyz[0] - x0)/fabs(x1 - x0);
            Real yd = fabs(apxyz[1] - y0)/fabs(y1 - y0);
            Real zd = fabs(apxyz[2] - z0)/fabs(z1 - z0);
            Int idx0 = IDX(i0, j0, k0, shape);
            Int idx1 = IDX(i1, j0, k0, shape);
            Int idx2 = IDX(i0, j1, k0, shape);
            Int idx3 = IDX(i1, j1, k0, shape);
            Int idx4 = IDX(i0, j0, k1, shape);
            Int idx5 = IDX(i1, j0, k1, shape);
            Int idx6 = IDX(i0, j1, k1, shape);
            Int idx7 = IDX(i1, j1, k1, shape);

            Real c0, c1, c2, c3, c4, c5, c6, c7;
            const Int dim_u = 0;
            c0 = uvw(idx0, dim_u);
            c1 = uvw(idx1, dim_u);
            c2 = uvw(idx2, dim_u);
            c3 = uvw(idx3, dim_u);
            c4 = uvw(idx4, dim_u);
            c5 = uvw(idx5, dim_u);
            c6 = uvw(idx6, dim_u);
            c7 = uvw(idx7, dim_u);
            Real u_at_ap = trilinear_interpolation(xd, yd, zd, c0, c1, c2, c3, c4, c5, c6, c7);
            const Int dim_v = 1;
            c0 = uvw(idx0, dim_v);
            c1 = uvw(idx1, dim_v);
            c2 = uvw(idx2, dim_v);
            c3 = uvw(idx3, dim_v);
            c4 = uvw(idx4, dim_v);
            c5 = uvw(idx5, dim_v);
            c6 = uvw(idx6, dim_v);
            c7 = uvw(idx7, dim_v);
            Real v_at_ap = trilinear_interpolation(xd, yd, zd, c0, c1, c2, c3, c4, c5, c6, c7);
            const Int dim_w = 2;
            c0 = uvw(idx0, dim_w);
            c1 = uvw(idx1, dim_w);
            c2 = uvw(idx2, dim_w);
            c3 = uvw(idx3, dim_w);
            c4 = uvw(idx4, dim_w);
            c5 = uvw(idx5, dim_w);
            c6 = uvw(idx6, dim_w);
            c7 = uvw(idx7, dim_w);
            Real w_at_ap = trilinear_interpolation(xd, yd, zd, c0, c1, c2, c3, c4, c5, c6, c7);

            Real3 base = turbines.base[turbine_id];
            Real3 base_velocity = turbines.base_velocity[turbine_id];
            Real3 apxyz_tt = one_angle_frame_rotation(apxyz - base, angle, angle_type);
            Real3 uvw_at_ap_tt = one_angle_frame_rotation_dt(apxyz - base, Real3{{u_at_ap, v_at_ap, w_at_ap}} - base_velocity, angle, angular_velocity, angle_type);

            Real theta0 = (2*Pi/n_blade)*blade_id;
            // t = floormod(t, 2*Pi/tip);
            Real theta  = tip*t + theta0;

            Real ux_tt = uvw_at_ap_tt[0];
            Real ut_tt = tip*apr + uvw_at_ap_tt[1]*sin(theta) - uvw_at_ap_tt[2]*cos(theta);
            Real urel2 = ux_tt*ux_tt + ut_tt*ut_tt;
            Real phi = atan(ux_tt/ut_tt);
            Real chord, twist, cl, cd;
            aps.get_airfoil_params(ap_id, rad2deg(phi), chord, twist, cl, cd);

            // printf("%lf\n", dr_per_ap);
            Real fl = .5*cl*urel2*chord*dr_per_ap;
            Real fd = .5*cd*urel2*chord*dr_per_ap;
            Real fx = fl*cos(phi) + fd*sin(phi);
            Real ft = fl*sin(phi) - fd*sin(phi);
            ft *= sign(tip);
            Real3 ff_tt{{-fx, ft*sin(theta), -ft*cos(theta)}};
            aps.force[ap_id] = one_angle_frame_rotation(ff_tt, - angle, angle_type);
            aps.torque[ap_id] = fabs(ft)*apr;
            aps.thrust[ap_id] = fx;
        } else {
            aps.force[ap_id][0] = 0;
            aps.force[ap_id][1] = 0;
            aps.force[ap_id][2] = 0;
            aps.torque[ap_id] = 0;
            aps.thrust[ap_id] = 0;
        }
    }
}

__global__ void kernel_DistributeAPForce(
    MatrixFrame<Real> *vx,
    MatrixFrame<Real> *vy,
    MatrixFrame<Real> *vz,
    MatrixFrame<Real> *vff,
    APFrame *vaps,
    Real euler_eps,
    Int3 shape,
    Int gc
) {
    const MatrixFrame<Real> &x = *vx;
    const MatrixFrame<Real> &y = *vy;
    const MatrixFrame<Real> &z = *vz;
    MatrixFrame<Real> &ff = *vff;
    APFrame &aps = *vaps;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0]-2*gc && j < shape[1]-2*gc && k < shape[2]-2*gc) {
        i += gc; j += gc; k += gc;
        Real cx = x(i);
        Real cy = y(j);
        Real cz = z(k);
        Real ffx = 0.;
        Real ffy = 0.;
        Real ffz = 0.;
        Real eta = 1./cubic(euler_eps*sqrt(Pi));
        for (Int ap_id = 0; ap_id < aps.apcount; ap_id ++) {
            const Real3 apxyz = aps.xyz[ap_id];
            Real rr2 = square(cx - apxyz[0]) + square(cy - apxyz[1]) + square(cz - apxyz[2]);

            ffx += aps.force[ap_id][0]*eta*exp(-rr2/square(euler_eps));
            ffy += aps.force[ap_id][1]*eta*exp(-rr2/square(euler_eps));
            ffz += aps.force[ap_id][2]*eta*exp(-rr2/square(euler_eps));
        }
        Int cid = IDX(i, j, k, shape);
        ff(cid, 0) = ffx;
        ff(cid, 1) = ffy;
        ff(cid, 2) = ffz;
    }
}

__global__ void kernel_DryDistribution(
    MatrixFrame<Real> *vx,
    MatrixFrame<Real> *vy,
    MatrixFrame<Real> *vz,
    MatrixFrame<Real> *vphi,
    APFrame *vaps,
    Real euler_eps,
    Int3 shape,
    Int gc
) {
    const MatrixFrame<Real> &x = *vx;
    const MatrixFrame<Real> &y = *vy;
    const MatrixFrame<Real> &z = *vz;
    MatrixFrame<Real> &phi = *vphi;
    APFrame &aps = *vaps;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0]-2*gc && j < shape[1]-2*gc && k < shape[2]-2*gc) {
        i += gc; j += gc; k += gc;
        Real cx = x(i);
        Real cy = y(j);
        Real cz = z(k);
        Real cphi = 0;
        Real eta = 1./cubic(euler_eps*sqrt(Pi));
        for (Int ap_id = 0; ap_id < aps.apcount; ap_id ++) {
            const Real3 apxyz = aps.xyz[ap_id];
            Real rr2 = square(cx - apxyz[0]) + square(cy - apxyz[1]) + square(cz - apxyz[2]);
            cphi += eta*exp(-rr2/square(euler_eps));
        }
        phi(IDX(i,j,k,shape)) = cphi;
    }
}

void AlmDevCall::UpdateTurbineAngles(Real t, size_t block_size) {
    size_t block_number = (turbines.n_turbine + block_size - 1)/block_size;
    kernel_UpdateTurbineAngles<<<block_number, block_size>>>(turbines.devptr, t);
}

void AlmDevCall::UpdateAPX(Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z, Real t, size_t block_size) {
    size_t block_number = (aps.apcount + block_size - 1)/block_size;
    kernel_UpdateAPX<<<block_number, block_size>>>(x.devptr, y.devptr, z.devptr, turbines.devptr, aps.devptr, n_ap_per_blade, t, gc, rank);
}

void AlmDevCall::CalcAPForce(Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z, Matrix<Real> &uvw, Real t, size_t block_size) {
    size_t block_number = (aps.apcount + block_size - 1)/block_size;
    kernel_CalcAPForce<<<block_number, block_size>>>(x.devptr, y.devptr, z.devptr, uvw.devptr, turbines.devptr, aps.devptr, n_ap_per_blade, pdm_shape, t, rank);
}

void AlmDevCall::DistributeAPForce(Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z, Matrix<Real> &ff, Real euler_eps, dim3 block_size) {
    dim3 block_number(
        (pdm_shape[0] + block_size.x - 1) / block_size.x,
        (pdm_shape[1] + block_size.y - 1) / block_size.y,
        (pdm_shape[2] + block_size.z - 1) / block_size.z
    );
    kernel_DistributeAPForce<<<block_number, block_size>>>(x.devptr, y.devptr, z.devptr, ff.devptr, aps.devptr, euler_eps, pdm_shape, gc);
}

void AlmDevCall::DryDistribution(Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z, Matrix<Real> &phi, Real euler_eps, dim3 block_size) {
    dim3 block_number(
        (pdm_shape[0] + block_size.x - 1) / block_size.x,
        (pdm_shape[1] + block_size.y - 1) / block_size.y,
        (pdm_shape[2] + block_size.z - 1) / block_size.z
    );
    kernel_DryDistribution<<<block_number, block_size>>>(x.devptr, y.devptr, z.devptr, phi.devptr, aps.devptr, euler_eps, pdm_shape, gc);
}

void AlmDevCall::CalcTorqueAndThrust() {
    int n_ap_per_turbine = n_ap_per_blade*turbines.n_blade;
    for (int tid = 0; tid < turbines.n_turbine; tid ++) {
        thrust::device_ptr<Real> torque_ptr = thrust::device_pointer_cast(aps.dev.torque + n_ap_per_turbine*tid);
        thrust::device_ptr<Real> thrust_ptr = thrust::device_pointer_cast(aps.dev.thrust + n_ap_per_turbine*tid);
        turbines.host.torque[tid] = thrust::reduce(torque_ptr, torque_ptr + n_ap_per_turbine);
        turbines.host.thrust[tid] = thrust::reduce(thrust_ptr, thrust_ptr + n_ap_per_turbine);
    }
}

}

}