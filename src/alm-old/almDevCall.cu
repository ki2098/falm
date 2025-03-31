#include "almDevCall.h"
#include "../dev/devutil.cuh"
#include "../falmath.h"

namespace Falm {

__device__ inline size_t find_position(Real *x, size_t xlen, Real v) {
    for (size_t i = 0; i < xlen-1; i ++) {
        if (x[i] <= v && x[i+1] >= v) {
            return i;
        }
    }
    return xlen-1;
}

__device__ inline size_t find_division(Int *offset, size_t ndivision, Int id) {
    for (size_t i = 0; i < ndivision-1; i ++) {
        if (offset[i] <= id && offset[i+1] > id) {
            return i;
        }
    }
    return ndivision-1;
}

__device__ inline Real trilinear_interpolate(
    Real xd,
    Real yd,
    Real zd,
    Real c0,
    Real c1,
    Real c2,
    Real c3,
    Real c4,
    Real c5,
    Real c6,
    Real c7
) {
    Real c01   = c0 *(1. - xd) + c1 *xd;
    Real c23   = c2 *(1. - xd) + c3 *xd;
    Real c45   = c4 *(1. - xd) + c5 *xd;
    Real c67   = c6 *(1. - xd) + c7 *xd;

    Real c0123 = c01*(1. - yd) + c23*yd;
    Real c4567 = c45*(1. - yd) + c67*yd;

    return c0123*(1. - zd) + c4567*zd;
}

__global__ void kernel_UpdateAPX(
    const MatrixFrame<Real> *vapx,
    const MatrixFrame<Real> *vapr,
    const MatrixFrame<Real> *vapTh,
    const MatrixFrame<int>  *vapTurbineIdx,
    const MatrixFrame<int>  *vaprank,
    const MatrixFrame<Int>  *vaplocal,
    const MatrixFrame<Real> *vturbineX,
    const MatrixFrame<Real> *vturbineRCXT,
    const MatrixFrame<Real> *vturbinePitch,
    const MatrixFrame<Real> *vturbineTipRate,
    const MatrixFrame<Real> *vx,
    const MatrixFrame<Real> *vy,
    const MatrixFrame<Real> *vz,
    const MatrixFrame<Int>  *vIoffset,
    const MatrixFrame<Int>  *vJoffset,
    const MatrixFrame<Int>  *vKoffset,
    int n_blades_per_turbine,
    int n_aps_per_blade,
    Real tt,
    Int3 mpi_shape
) {
    const MatrixFrame<Real> &apx = *vapx;
    const MatrixFrame<Real> &apr = *vapr;
    const MatrixFrame<Real> &apth = *vapTh;
    const MatrixFrame<int>  &aptidx = *vapTurbineIdx;
    const MatrixFrame<int>  &aprank = *vaprank;
    const MatrixFrame<Int>  &aplocal = *vaplocal;
    const MatrixFrame<Real> &turbinex = *vturbineX;
    const MatrixFrame<Real> &rcxt = *vturbineRCXT;
    const MatrixFrame<Real> &pitch = *vturbinePitch;
    const MatrixFrame<Real> &tip = *vturbineTipRate;
    const MatrixFrame<Real> &x = *vx, &y = *vy, &z = *vz;
    const MatrixFrame<Int>  &Ioffset = *vIoffset, &Joffset = *vJoffset, &Koffset = *vKoffset;
    int apid = GLOBAL_THREAD_IDX();
    if (apid < apx.shape[0]) {
        int ap_turbine_id = aptidx(apid);
        int n_aps_per_turbine = n_blades_per_turbine*n_aps_per_blade;
        int turbine_ap_id = apid % n_aps_per_turbine;
        int ap_blade_id = turbine_ap_id / n_aps_per_blade;
        Real initial_theta = ap_blade_id*2*Pi/n_blades_per_turbine + 0.5*Pi;
        Real theta = initial_theta + tip(ap_turbine_id)*tt;
        apth(apid) = theta;

        Real R = apr(apid);
        Real3 apvect = rcxt(ap_turbine_id) + Real3{{0., R*cos(theta), R*sin(theta)}};
        Real3 apvec = pitch_only_vector_turbine2abs(apvect, pitch(ap_turbine_id));
        Real apX = turbinex(ap_turbine_id, 0) + apvec[0];
        Real apY = turbinex(ap_turbine_id, 1) + apvec[1];
        Real apZ = turbinex(ap_turbine_id, 2) + apvec[2];
        Int  apI = find_position(x.ptr, x.size, apX);
        Int  apJ = find_position(y.ptr, y.size, apY);
        Int  apK = find_position(z.ptr, z.size, apZ);
        
        apx(apid, 0) = apX;
        apx(apid, 1) = apY;
        apx(apid, 2) = apZ;
        aplocal(apid, 0) = apI;
        aplocal(apid, 1) = apJ;
        aplocal(apid, 2) = apK;
        
        Int diviI = find_division(Ioffset.ptr, Ioffset.size, apI);
        Int diviJ = find_division(Joffset.ptr, Joffset.size, apJ);
        Int diviK = find_division(Koffset.ptr, Koffset.size, apK);
        aprank(apid) = IDX(diviI, diviJ, diviK, mpi_shape);
    }
    
}

__global__ void kernel_CalcAPForce(
    const MatrixFrame<Real> *vapx,
    const MatrixFrame<Real> *vapr,
    const MatrixFrame<Real> *vapTh,
    const MatrixFrame<Real> *vapChord,
    const MatrixFrame<Real> *vapTwist,
    const MatrixFrame<Real> *vapf,
    const MatrixFrame<int>  *vapTurbineIdx,
    const MatrixFrame<int>  *vaprank,
    const MatrixFrame<Int>  *vaplocal,
    const MatrixFrame<Real> *vturbineX,
    const MatrixFrame<Real> *vturbinePitch,
    const MatrixFrame<Real> *vturbinePitchRate,
    const MatrixFrame<Real> *vturbineTipRate,
    const MatrixFrame<Real> *vu,
    const MatrixFrame<Real> *vx,
    const MatrixFrame<Real> *vy,
    const MatrixFrame<Real> *vz,
    Real dr_per_ap,
    Int3 shape,
    int rank
) {
    const MatrixFrame<Real> &apx = *vapx;
    const MatrixFrame<Real> &apr = *vapr;
    const MatrixFrame<Real> &apTh = *vapTh;
    const MatrixFrame<Real> &apChord = *vapChord;
    const MatrixFrame<Real> &apTwist = *vapTwist;
    const MatrixFrame<Real> &apf = *vapf;
    const MatrixFrame<int>  &apTurbineIdx = *vapTurbineIdx;
    const MatrixFrame<int>  &aprank = *vaprank;
    const MatrixFrame<Int>  &aplocal = *vaplocal;
    const MatrixFrame<Real> &turbineX = *vturbineX;
    const MatrixFrame<Real> &turbinePitch = *vturbinePitch;
    const MatrixFrame<Real> &turbinePitchRate = *vturbinePitchRate;
    const MatrixFrame<Real> &turbineTipRate = *vturbineTipRate;
    const MatrixFrame<Real> &u = *vu;
    const MatrixFrame<Real> &x = *vx;
    const MatrixFrame<Real> &y = *vy;
    const MatrixFrame<Real> &z = *vz;

    Int apid = GLOBAL_THREAD_IDX();
    if (apid < apx.shape[0]) {
        if (aprank(apid) == rank) {
            int turbineidx = apTurbineIdx(apid);
            Real pitch = turbinePitch(turbineidx);
            Real pitchRate = turbinePitchRate(turbineidx);
            Int I = aplocal(apid, 0), J = aplocal(apid, 1), K = aplocal(apid, 2);
            Real apX = apx(apid, 0), apY = apx(apid, 1), apZ = apx(apid, 2);
            Real X0 = x(I), X1 = x(I+1);
            Real Y0 = y(J), Y1 = y(J+1);
            Real Z0 = z(K), Z1 = z(K+1);
            Real xd = fabs(apX - X0) / fabs(X1 - X0);
            Real yd = fabs(apY - Y0) / fabs(Y1 - Y0);
            Real zd = fabs(apZ - Z0) / fabs(Z1 - Z0);
            Int idx1 = IDX(I  ,J  ,K  ,shape);
            Int idx2 = IDX(I+1,J  ,K  ,shape);
            Int idx3 = IDX(I  ,J+1,K  ,shape);
            Int idx4 = IDX(I+1,J+1,K  ,shape);
            Int idx5 = IDX(I  ,J  ,K+1,shape);
            Int idx6 = IDX(I+1,J  ,K+1,shape);
            Int idx7 = IDX(I  ,J+1,K+1,shape);
            Int idx8 = IDX(I+1,J+1,K+1,shape);
            Real u1 = u(idx1, 0);
            Real u2 = u(idx2, 0);
            Real u3 = u(idx3, 0);
            Real u4 = u(idx4, 0);
            Real u5 = u(idx5, 0);
            Real u6 = u(idx6, 0);
            Real u7 = u(idx7, 0);
            Real u8 = u(idx8, 0);
            Real v1 = u(idx1, 1);
            Real v2 = u(idx2, 1);
            Real v3 = u(idx3, 1);
            Real v4 = u(idx4, 1);
            Real v5 = u(idx5, 1);
            Real v6 = u(idx6, 1);
            Real v7 = u(idx7, 1);
            Real v8 = u(idx8, 1);
            Real w1 = u(idx1, 2);
            Real w2 = u(idx2, 2);
            Real w3 = u(idx3, 2);
            Real w4 = u(idx4, 2);
            Real w5 = u(idx5, 2);
            Real w6 = u(idx6, 2);
            Real w7 = u(idx7, 2);
            Real w8 = u(idx8, 2);
            Real uc = trilinear_interpolate(xd, yd, zd, u1, u2, u3, u4, u5, u6, u7, u8);
            Real vc = trilinear_interpolate(xd, yd, zd, v1, v2, v3, v4, v5, v6, v7, v8);
            Real wc = trilinear_interpolate(xd, yd, zd, w1, w2, w3, w4, w5, w6, w7, w8);
            
            Real3 _x0{{turbineX(turbineidx, 0), turbineX(turbineidx, 1), turbineX(turbineidx, 2)}};
            Real3 _x{{apX, apY, apZ}};
            Real3 _u{{uc, vc, wc}};
            Real3 UatAP = pitch_only_velocity_abs2turbine(_u, _x, _x0, pitch, pitchRate);
            Real uXT = UatAP[0];

            Real tip = turbineTipRate(turbineidx);
            Real theta = apTh(apid);
            Real uTT = sign(tip)*(apr(apid)*tip + UatAP[1]*sin(theta) - UatAP[2]*cos(theta));

            Real uRel2 = square(uXT) + square(uTT);
            Real phi = atan2(uXT, uTT);
            Real attack = phi*180./Pi - apTwist(apid);
            Real chord = apChord(apid);

            Real fl = 0.5*clfunc(attack)*uRel2*chord*dr_per_ap;
            Real fd = 0.5*cdfunc(attack)*uRel2*chord*dr_per_ap;
            Real fx = fl*cos(phi) + fd*sin(phi);
            Real ft = fl*sin(phi) - fd*cos(phi);
            ft *= sign(tip);
            Real3 ff = pitch_only_vector_turbine2abs({{-fx, ft*sin(theta), -ft*cos(theta)}}, pitch);
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
    const MatrixFrame<Real> *vapx,
    const MatrixFrame<Real> *vapf,
    const MatrixFrame<Real> *vx,
    const MatrixFrame<Real> *vy,
    const MatrixFrame<Real> *vz,
    const MatrixFrame<Real> *vff,
    Real euler_eps,
    Int3 shape,
    Int gc
) {
    const MatrixFrame<Real> &apx=*vapx, &apf=*vapf, &x=*vx, &y=*vy, &z=*vz, &ff=*vff;
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
        Real eta = 1./(cubic(euler_eps)*cbrt(square(Pi)));
        for (int apid = 0; apid < apx.size; apid ++) {
            Real ax = apx(apid, 0);
            Real ay = apx(apid, 1);
            Real az = apx(apid, 2);
            Real rr2 = square(cx-ax)+square(cy-ay)+square(cz-az);
            
            ffx += apf(apid, 0)*eta*exp(-rr2/square(euler_eps));
            ffy += apf(apid, 1)*eta*exp(-rr2/square(euler_eps));
            ffz += apf(apid, 2)*eta*exp(-rr2/square(euler_eps));
        }
        Int cidx = IDX(i, j, k, shape);
        ff(cidx, 0) = ffx;
        ff(cidx, 1) = ffy;
        ff(cidx, 2) = ffz;
    }
}

}