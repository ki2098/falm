#include "almDevCall.h"
#include "../dev/devutil.cuh"
#include "../falmath.h"

namespace Falm {

__device__ inline size_t find_position(REAL *x, size_t xlen, REAL v) {
    for (size_t i = 0; i < xlen-1; i ++) {
        if (x[i] <= v && x[i+1] >= v) {
            return i;
        }
    }
    return xlen-1;
}

__device__ inline size_t find_division(INT *offset, size_t ndivision, INT id) {
    for (size_t i = 0; i < ndivision-1; i ++) {
        if (offset[i] <= id && offset[i+1] > id) {
            return i;
        }
    }
    return ndivision-1;
}

__device__ inline REAL trilinear_interpolate(
    REAL xd,
    REAL yd,
    REAL zd,
    REAL c0,
    REAL c1,
    REAL c2,
    REAL c3,
    REAL c4,
    REAL c5,
    REAL c6,
    REAL c7
) {
    REAL c01   = c0 *(1. - xd) + c1 *xd;
    REAL c23   = c2 *(1. - xd) + c3 *xd;
    REAL c45   = c4 *(1. - xd) + c5 *xd;
    REAL c67   = c6 *(1. - xd) + c7 *xd;

    REAL c0123 = c01*(1. - yd) + c23*yd;
    REAL c4567 = c45*(1. - yd) + c67*yd;

    return c0123*(1. - zd) + c4567*zd;
}

__global__ void kernel_UpdateAPX(
    const MatrixFrame<REAL> *vapx,
    const MatrixFrame<REAL> *vapr,
    const MatrixFrame<REAL> *vapTh,
    const MatrixFrame<int>  *vapTurbineIdx,
    const MatrixFrame<int>  *vaprank,
    const MatrixFrame<INT>  *vaplocal,
    const MatrixFrame<REAL> *vturbineX,
    const MatrixFrame<REAL> *vturbineRCXT,
    const MatrixFrame<REAL> *vturbinePitch,
    const MatrixFrame<REAL> *vturbineTipRate,
    const MatrixFrame<REAL> *vx,
    const MatrixFrame<REAL> *vy,
    const MatrixFrame<REAL> *vz,
    const MatrixFrame<INT>  *vIoffset,
    const MatrixFrame<INT>  *vJoffset,
    const MatrixFrame<INT>  *vKoffset,
    int n_blades_per_turbine,
    int n_aps_per_blade,
    REAL tt,
    INT3 mpi_shape
) {
    const MatrixFrame<REAL> &apx = *vapx;
    const MatrixFrame<REAL> &apr = *vapr;
    const MatrixFrame<REAL> &apth = *vapTh;
    const MatrixFrame<int>  &aptidx = *vapTurbineIdx;
    const MatrixFrame<int>  &aprank = *vaprank;
    const MatrixFrame<INT>  &aplocal = *vaplocal;
    const MatrixFrame<REAL> &turbinex = *vturbineX;
    const MatrixFrame<REAL> &rcxt = *vturbineRCXT;
    const MatrixFrame<REAL> &pitch = *vturbinePitch;
    const MatrixFrame<REAL> &tip = *vturbineTipRate;
    const MatrixFrame<REAL> &x = *vx, &y = *vy, &z = *vz;
    const MatrixFrame<INT>  &Ioffset = *vIoffset, &Joffset = *vJoffset, &Koffset = *vKoffset;
    int apid = GLOBAL_THREAD_IDX();
    if (apid < apx.shape[0]) {
        int ap_turbine_id = aptidx(apid);
        int n_aps_per_turbine = n_blades_per_turbine*n_aps_per_blade;
        int turbine_ap_id = apid % n_aps_per_turbine;
        int ap_blade_id = turbine_ap_id / n_aps_per_blade;
        REAL initial_theta = ap_blade_id*2*Pi/n_blades_per_turbine + 0.5*Pi;
        REAL theta = initial_theta + tip(ap_turbine_id)*tt;
        apth(apid) = theta;

        REAL R = apr(apid);
        REAL3 apvect = rcxt(apid) + REAL3{{0., R*cos(theta), R*sin(theta)}};
        REAL3 apvec = pitch_only_vector_turbine2abs(apvect, pitch(ap_turbine_id));
        REAL apX = turbinex(ap_turbine_id, 0) + apvec[0];
        REAL apY = turbinex(ap_turbine_id, 1) + apvec[1];
        REAL apZ = turbinex(ap_turbine_id, 2) + apvec[2];
        INT  apI = find_position(x.ptr, x.size, apX);
        INT  apJ = find_position(y.ptr, y.size, apY);
        INT  apK = find_position(z.ptr, z.size, apZ);
        
        apx(apid, 0) = apX;
        apx(apid, 1) = apY;
        apx(apid, 2) = apZ;
        aplocal(apid, 0) = apI;
        aplocal(apid, 1) = apJ;
        aplocal(apid, 2) = apK;
        
        INT diviI = find_division(Ioffset.ptr, Ioffset.size, apI);
        INT diviJ = find_division(Joffset.ptr, Joffset.size, apJ);
        INT diviK = find_division(Koffset.ptr, Koffset.size, apK);
        aprank(apid) = IDX(diviI, diviJ, diviK, mpi_shape);
    }
    
}

__global__ void kernel_CalcAPForce(
    const MatrixFrame<REAL> *vapx,
    const MatrixFrame<REAL> *vapr,
    const MatrixFrame<REAL> *vapTh,
    const MatrixFrame<REAL> *vapChord,
    const MatrixFrame<REAL> *vapTwist,
    const MatrixFrame<REAL> *vapf,
    const MatrixFrame<int>  *vapTurbineIdx,
    const MatrixFrame<int>  *vaprank,
    const MatrixFrame<INT>  *vaplocal,
    const MatrixFrame<REAL> *vturbineX,
    const MatrixFrame<REAL> *vturbinePitch,
    const MatrixFrame<REAL> *vturbinePitchRate,
    const MatrixFrame<REAL> *vturbineTipRate,
    const MatrixFrame<REAL> *vu,
    const MatrixFrame<REAL> *vx,
    const MatrixFrame<REAL> *vy,
    const MatrixFrame<REAL> *vz,
    REAL dr_per_ap,
    INT3 shape,
    int rank
) {
    const MatrixFrame<REAL> &apx = *vapx;
    const MatrixFrame<REAL> &apr = *vapr;
    const MatrixFrame<REAL> &apTh = *vapTh;
    const MatrixFrame<REAL> &apChord = *vapChord;
    const MatrixFrame<REAL> &apTwist = *vapTwist;
    const MatrixFrame<REAL> &apf = *vapf;
    const MatrixFrame<int>  &apTurbineIdx = *vapTurbineIdx;
    const MatrixFrame<int>  &aprank = *vaprank;
    const MatrixFrame<INT>  &aplocal = *vaplocal;
    const MatrixFrame<REAL> &turbineX = *vturbineX;
    const MatrixFrame<REAL> &turbinePitch = *vturbinePitch;
    const MatrixFrame<REAL> &turbinePitchRate = *vturbinePitchRate;
    const MatrixFrame<REAL> &turbineTipRate = *vturbineTipRate;
    const MatrixFrame<REAL> &u = *vu;
    const MatrixFrame<REAL> &x = *vx;
    const MatrixFrame<REAL> &y = *vy;
    const MatrixFrame<REAL> &z = *vz;

    INT apid = GLOBAL_THREAD_IDX();
    if (apid < apx.shape[0]) {
        if (aprank(apid) == rank) {
            int turbineidx = apTurbineIdx(apid);
            REAL pitch = turbinePitch(turbineidx);
            REAL pitchRate = turbinePitchRate(turbineidx);
            INT I = aplocal(apid, 0), J = aplocal(apid, 1), K = aplocal(apid, 2);
            REAL apX = apx(apid, 0), apY = apx(apid, 1), apZ = apx(apid, 2);
            REAL X0 = x(I), X1 = x(I+1);
            REAL Y0 = y(J), Y1 = y(J+1);
            REAL Z0 = z(K), Z1 = z(K+1);
            REAL xd = fabs(apX - X0) / fabs(X1 - X0);
            REAL yd = fabs(apY - Y0) / fabs(Y1 - Y0);
            REAL zd = fabs(apZ - Z0) / fabs(Z1 - Z0);
            INT idx1 = IDX(I  ,J  ,K  ,shape);
            INT idx2 = IDX(I+1,J  ,K  ,shape);
            INT idx3 = IDX(I  ,J+1,K  ,shape);
            INT idx4 = IDX(I+1,J+1,K  ,shape);
            INT idx5 = IDX(I  ,J  ,K+1,shape);
            INT idx6 = IDX(I+1,J  ,K+1,shape);
            INT idx7 = IDX(I  ,J+1,K+1,shape);
            INT idx8 = IDX(I+1,J+1,K+1,shape);
            REAL u1 = u(idx1, 0);
            REAL u2 = u(idx2, 0);
            REAL u3 = u(idx3, 0);
            REAL u4 = u(idx4, 0);
            REAL u5 = u(idx5, 0);
            REAL u6 = u(idx6, 0);
            REAL u7 = u(idx7, 0);
            REAL u8 = u(idx8, 0);
            REAL v1 = u(idx1, 1);
            REAL v2 = u(idx2, 1);
            REAL v3 = u(idx3, 1);
            REAL v4 = u(idx4, 1);
            REAL v5 = u(idx5, 1);
            REAL v6 = u(idx6, 1);
            REAL v7 = u(idx7, 1);
            REAL v8 = u(idx8, 1);
            REAL w1 = u(idx1, 2);
            REAL w2 = u(idx2, 2);
            REAL w3 = u(idx3, 2);
            REAL w4 = u(idx4, 2);
            REAL w5 = u(idx5, 2);
            REAL w6 = u(idx6, 2);
            REAL w7 = u(idx7, 2);
            REAL w8 = u(idx8, 2);
            REAL uc = trilinear_interpolate(xd, yd, zd, u1, u2, u3, u4, u5, u6, u7, u8);
            REAL vc = trilinear_interpolate(xd, yd, zd, v1, v2, v3, v4, v5, v6, v7, v8);
            REAL wc = trilinear_interpolate(xd, yd, zd, w1, w2, w3, w4, w5, w6, w7, w8);
            
            REAL3 _x0{{turbineX(turbineidx, 0), turbineX(turbineidx, 1), turbineX(turbineidx, 2)}};
            REAL3 _x{{apX, apY, apZ}};
            REAL3 _u{{uc, vc, wc}};
            REAL3 UatAP = pitch_only_velocity_abs2turbine(_u, _x, _x0, pitch, pitchRate);
            REAL uXT = UatAP[0];

            REAL tip = turbineTipRate(turbineidx);
            REAL theta = apTh(apid);
            REAL uTT = sign(tip)*(apr(apid)*tip + UatAP[1]*sin(theta) - UatAP[2]*cos(theta));

            REAL uRel2 = square(uXT) + square(uTT);
            REAL phi = atan2(uXT, uTT);
            REAL attack = phi*180./Pi - apTwist(apid);
            REAL chord = apChord(apid);

            REAL fl = 0.5*clfunc(attack)*uRel2*chord*dr_per_ap;
            REAL fd = 0.5*cdfunc(attack)*uRel2*chord*dr_per_ap;
            REAL fx = fl*cos(phi) + fd*sin(phi);
            REAL ft = fl*sin(phi) - fd*cos(phi);
            ft *= sign(tip);
            REAL3 ff = pitch_only_vector_turbine2abs({{-fx, ft*sin(theta), -ft*cos(theta)}}, pitch);
            apf(apid, 0) = ff[0];
            apf(apid, 1) = ff[1];
            apf(apid, 2) = ff[2];
        } else {
            apf(apid, 0) = 0.;
            apf(apid, 1) = 0.;
            apf(apid, 2) = 0.;
        }
    }
}

__global__ void kernel_APForceDistribute(
    const MatrixFrame<REAL> *vapx,
    const MatrixFrame<REAL> *vapf,
    const MatrixFrame<REAL> *vx,
    const MatrixFrame<REAL> *vy,
    const MatrixFrame<REAL> *vz,
    const MatrixFrame<REAL> *vff,
    REAL euler_eps,
    INT3 shape,
    INT gc
) {
    const MatrixFrame<REAL> &apx=*vapx, &apf=*vapf, &x=*vx, &y=*vy, &z=*vz, &ff=*vff;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0]-2*gc && j < shape[1]-2*gc && k < shape[2]-2*gc) {
        i += gc; j += gc; k += gc;
        REAL cx = x(i);
        REAL cy = y(j);
        REAL cz = z(k);
        REAL ffx = 0.;
        REAL ffy = 0.;
        REAL ffz = 0.;
        REAL eta = 1./(cube(euler_eps)*cbrt(square(Pi)));
        for (int apid = 0; apid < apx.size; apid ++) {
            REAL ax = apx(apid, 0);
            REAL ay = apx(apid, 1);
            REAL az = apx(apid, 2);
            REAL rr2 = square(cx-ax)+square(cy-ay)+square(cz-az);
            
            ffx += apf(apid, 0)*eta*exp(-rr2/square(euler_eps));
            ffy += apf(apid, 1)*eta*exp(-rr2/square(euler_eps));
            ffz += apf(apid, 2)*eta*exp(-rr2/square(euler_eps));
        }
        INT cidx = IDX(i, j, k, shape);
        ff(cidx, 0) = ffx;
        ff(cidx, 1) = ffy;
        ff(cidx, 2) = ffz;
    }
}

}