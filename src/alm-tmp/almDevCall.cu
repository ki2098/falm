#include "almDevCall.h"
#include "../dev/devutil.cuh"
#include "../falmath.h"

namespace Falm {

template<typename T>
__host__ __device__ size_t find_interval(T *x, size_t size, T p) {
    for (size_t i = 0; i < size-1; i ++) {
        if (x[i] <= p && x[i+1] > p) {
            return i;
        }
    }
    return size-1;
}

__host__ __device__ inline Real trilinear_interpolate(
    Real xp, Real yp, Real zp,
    Real a0, Real a1, Real a2, Real a3, Real a4, Real a5, Real a6, Real a7
) {
    Real a8  = (1 - xp)*a0  + xp*a1;
    Real a9  = (1 - xp)*a2  + xp*a3;
    Real a10 = (1 - xp)*a4  + xp*a5;
    Real a11 = (1 - xp)*a6  + xp*a7;
    Real a12 = (1 - yp)*a8  + yp*a9;
    Real a13 = (1 - yp)*a10 + yp*a11;
    return (1 - zp)*a12 + zp*a13;
}

__global__ void kernel_updateApx(
    MatrixFrame<Real> *vapx,
    MatrixFrame<Int>  *vapi,
    MatrixFrame<Real> *vapr,
    MatrixFrame<Real> *vaptheta,
    MatrixFrame<int>  *vaptid,
    MatrixFrame<int>  *vapbid,
    MatrixFrame<int>  *vaprank,
    MatrixFrame<Real> *vfoundx,
    MatrixFrame<Real> *vhubx,
    MatrixFrame<Real> *vpitch,
    MatrixFrame<Real> *vtiprate,
    MatrixFrame<Real> *vx,
    MatrixFrame<Real> *vy,
    MatrixFrame<Real> *vz,
    MatrixFrame<Int>  *vioffset,
    MatrixFrame<Int>  *vjoffset,
    MatrixFrame<Int>  *vkoffset,
    int nbpt,
    int nappb,
    Real tt,
    Int3 mpi_shape
) {
    MatrixFrame<Real> &apx = *vapx;
    MatrixFrame<Int> &api = *vapi;
    MatrixFrame<Real> &apr = *vapr;
    MatrixFrame<Real> &aptheta = *vaptheta;
    MatrixFrame<int>  &aptid = *vaptid;
    MatrixFrame<int>  &apbid = *vapbid;
    MatrixFrame<int>  &aprank = *vaprank;
    MatrixFrame<Real> &foundx = *vfoundx;
    MatrixFrame<Real> &hubx = *vhubx;
    MatrixFrame<Real> &pitch = *vpitch;
    MatrixFrame<Real> &tiprate = *vtiprate;
    MatrixFrame<Real> &x = *vx, &y = *vy, &z = *vz;
    MatrixFrame<Int> &ioffset = *vioffset, &joffset = *vjoffset, &koffset = *vkoffset;
    int apid = GLOBAL_THREAD_IDX();
    if (apid < apx.shape[0]) {
        int tid = aptid(apid);
        Real initial_theta = apbid(apid)*2.*Pi/nbpt + 0.5*Pi;
        Real theta = initial_theta + tiprate(tid)*tt;
        aptheta(apid) = theta;

        Real rt = apr(apid);
        Real3 apvt = hubx(tid) + Real3{{0., rt*cos(theta), rt*sin(theta)}};
        Real3 apv  = pitch_only_vector_turbine2abs(apvt, pitch(tid));
        Real apX = foundx(tid, 0) + apv[0];
        Real apY = foundx(tid, 1) + apv[1];
        Real apZ = foundx(tid, 2) + apv[2];
        Int  apI = find_interval(x.ptr, x.size, apX);
        Int  apJ = find_interval(y.ptr, y.size, apY);
        Int  apK = find_interval(z.ptr, z.size, apZ);

        apx(apid, 0) = apX;
        apx(apid, 1) = apY;
        apx(apid, 2) = apZ;
        api(apid, 0) = apI;
        api(apid, 1) = apJ;
        api(apid, 2) = apK;

        Int rankI = find_interval(ioffset.ptr, ioffset.size, apI);
        Int rankJ = find_interval(joffset.ptr, joffset.size, apJ);
        Int rankK = find_interval(koffset.ptr, koffset.size, apK);
        aprank(apid) = IDX(rankI, rankJ, rankK, mpi_shape);
    }
}

__global__ void kernel_calcApForce(
    MatrixFrame<Real> *vapx,
    MatrixFrame<Int>  *vapi,
    MatrixFrame<Real> *vapr,
    MatrixFrame<Real> *vaptheta,
    MatrixFrame<Real> *vapchord,
    MatrixFrame<Real> *vaptwist,
    MatrixFrame<Real> *vapcdcl,
    MatrixFrame<int>  *vaptid,
    MatrixFrame<int>  *vapbid,
    MatrixFrame<int>  *vaprank,
    MatrixFrame<Real> *vapff,
    MatrixFrame<Real> *vfoundx,
    MatrixFrame<Real> *vhubx,
    MatrixFrame<Real> *vpitch,
    MatrixFrame<Real> *vpitchrate,
    MatrixFrame<Real> *vtiprate,
    MatrixFrame<Real> *vu,
    MatrixFrame<Real> *vx,
    MatrixFrame<Real> *vy,
    MatrixFrame<Real> *vz,
    Real dr,
    Int3 shape,
    int rank
) {
    MatrixFrame<Real> &apx = *vapx;
    MatrixFrame<Int> &api = *vapi;
    MatrixFrame<Real> &apr = *vapr;
    MatrixFrame<Real> &aptheta = *vaptheta;
    MatrixFrame<Real> &apchord = *vapchord;
    MatrixFrame<Real> &aptwist = *vaptwist;
    MatrixFrame<Real> &apcdcl = *vapcdcl;
    MatrixFrame<int>  &aptid = *vaptid;
    MatrixFrame<int>  &apbid = *vapbid;
    MatrixFrame<int>  &aprank = *vaprank;
    MatrixFrame<Real> &apff = *vapff;
    MatrixFrame<Real> &foundx = *vfoundx;
    MatrixFrame<Real> &hubx = *vhubx;
    MatrixFrame<Real> &pitch = *vpitch;
    MatrixFrame<Real> &pitchrate = *vpitchrate;
    MatrixFrame<Real> &tiprate = *vtiprate;
    MatrixFrame<Real> &u = *vu, &x = *vx, &y = *vy, &z = *vz;

    Int apid = GLOBAL_THREAD_IDX();
    if (apid < apx.shape[0]) {
        if (aprank(apid) == rank) {
            int tid = aptid(apid);
            Real tpitch = pitch(tid);
            Real tpitchrate = pitchrate(tid);
            Int  apI = api(apid,0), apJ = api(apid,1), apK = api(apid,2);
            Real apX = apx(apid,0), apY = apx(apid,1), apZ = apx(apid,2);
            Real x0 = x(apI), x1 = x(apI+1);
            Real y0 = y(apJ), y1 = y(apJ+1);
            Real z0 = z(apK), z1 = z(apK+1);
            Real xp = (apX - x0)/(x1 - x0);
            Real yp = (apY - y0)/(y1 - y0);
            Real zp = (apZ - z0)/(z1 - z0);
            Int idx0 = IDX(apI  ,apJ  ,apK  ,shape);
            Int idx1 = IDX(apI+1,apJ  ,apK  ,shape);
            Int idx2 = IDX(apI  ,apJ+1,apK  ,shape);
            Int idx3 = IDX(apI+1,apJ+1,apK  ,shape);
            Int idx4 = IDX(apI  ,apJ  ,apK+1,shape);
            Int idx5 = IDX(apI+1,apJ  ,apK+1,shape);
            Int idx6 = IDX(apI  ,apJ+1,apK+1,shape);
            Int idx7 = IDX(apI+1,apJ+1,apK+1,shape);
            Real u0 = u(idx0, 0);
            Real u1 = u(idx1, 0);
            Real u2 = u(idx2, 0);
            Real u3 = u(idx3, 0);
            Real u4 = u(idx4, 0);
            Real u5 = u(idx5, 0);
            Real u6 = u(idx6, 0);
            Real u7 = u(idx7, 0);
            Real v0 = u(idx0, 1);
            Real v1 = u(idx1, 1);
            Real v2 = u(idx2, 1);
            Real v3 = u(idx3, 1);
            Real v4 = u(idx4, 1);
            Real v5 = u(idx5, 1);
            Real v6 = u(idx6, 1);
            Real v7 = u(idx7, 1);
            Real w0 = u(idx0, 2);
            Real w1 = u(idx1, 2);
            Real w2 = u(idx2, 2);
            Real w3 = u(idx3, 2);
            Real w4 = u(idx4, 2);
            Real w5 = u(idx5, 2);
            Real w6 = u(idx6, 2);
            Real w7 = u(idx7, 2);
            Real uc = trilinear_interpolate(xp, yp, zp, u0, u1, u2, u3, u4, u5, u6, u7);
            Real vc = trilinear_interpolate(xp, yp, zp, v0, v1, v2, v3, v4, v5, v6, v7);
            Real wc = trilinear_interpolate(xp, yp, zp, w0, w1, w2, w3, w4, w5, w6, w7);

            Real3 X0{{foundx(tid,0), foundx(tid,1), foundx(tid,2)}};
            Real3 X{{apX, apY, apZ}};
            Real3 U{{uc, vc, wc}};
            Real3 UAP = pitch_only_velocity_abs2turbine(U, X, X0, tpitch, tpitchrate);

            Real theta = aptheta(apid);
            Real ttiprate = tiprate(tid);
            Real utt = sign(ttiprate)*(apr(apid)*ttiprate + UAP[1]*sin(theta) - UAP[2]*cos(theta));
            Real uxt = UAP[0];
            Real urel2 = square(uxt) + square(utt);
            Real phi = atan2(uxt, utt);
            Real attack = phi*180./Pi - aptwist(apid);
            Real chord = apchord(apid);
            Real cd, cl;
            getcdcl(vapcdcl, apid, attack, cd, cl);
            Real fl = .5*cl*urel2*chord*dr;
            Real fd = .5*cd*urel2*chord*dr;
            Real fx = fl*cos(phi) + fd*sin(phi);
            Real ft = fl*sin(phi) - fd*cos(phi);
            ft *= sign(ttiprate);
            Real3 ff = pitch_only_vector_turbine2abs(Real3{{-fx,ft*sin(theta),-ft*cos(theta)}}, tpitch);
            apff(apid,0) = ff[0];
            apff(apid,1) = ff[1];
            apff(apid,2) = ff[2];
        } else {
            apff(apid,0) = 0;
            apff(apid,1) = 0;
            apff(apid,2) = 0;
        }
    }
}

__global__ void kernel_distributeForce(
    MatrixFrame<Real> *vapx,
    MatrixFrame<Real> *vapff,
    MatrixFrame<Real> *vx,
    MatrixFrame<Real> *vff,
    Real euler_eps,
    Int3 shape,
    Int3 offset
) {
    MatrixFrame<Real> &apx=*vapx, &apff=*vapff, &x=*vx, &ff=*vff;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] && j < shape[1] && k < shape[2]) {
        i += offset[0];
        j += offset[1];
        k += offset[2];
        Int cid = IDX(i,j,k,shape);
        Real cx = x(cid,0);
        Real cy = x(cid,1);
        Real cz = x(cid,2);
        Real ffx = 0, ffy = 0, ffz = 0;
        Real eta = 1./(cubic(euler_eps)*cbrt(square(Pi)));
        for (int apid = 0; apid < apff.shape[0]; apid ++) {
            Real ax = apx(apid, 0);
            Real ay = apx(apid, 1);
            Real az = apx(apid, 2);
            Real rr2 = square(cx-ax) + square(cy-ay) + square(cz-az);

            ffx += apff(apid,0)*eta*exp(-rr2/square(euler_eps));
            ffy += apff(apid,1)*eta*exp(-rr2/square(euler_eps));
            ffz += apff(apid,2)*eta*exp(-rr2/square(euler_eps));
        }
        ff(cid,0) = ffx;
        ff(cid,1) = ffy;
        ff(cid,2) = ffz;
    }
}

void FalmAlmDevCall::updateAp(
    APArray &aps,
    TurbineArray &turbines,
    Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z,
    Matrix<Real> &u,
    Real tt,
    CPM &cpm,
    int block_size
) {
    int grid_size = (aps.nap + block_size - 1)/block_size;
    kernel_updateApx<<<grid_size, block_size>>>(
        aps.apx.devptr,
        aps.api.devptr,
        aps.apr.devptr,
        aps.aptheta.devptr,
        aps.aptid.devptr,
        aps.apbid.devptr,
        aps.aprank.devptr,
        turbines.foundx.devptr,
        turbines.hubx.devptr,
        turbines.pitch.devptr,
        turbines.tiprate.devptr,
        x.devptr,
        y.devptr,
        z.devptr,
        ioffset.devptr,
        joffset.devptr,
        koffset.devptr,
        turbines.nbpt,
        turbines.nappb,
        tt,
        cpm.shape
    );
    kernel_calcApForce<<<grid_size, block_size>>>(
        aps.apx.devptr,
        aps.api.devptr,
        aps.apr.devptr,
        aps.aptheta.devptr,
        aps.apchord.devptr,
        aps.aptwist.devptr,
        aps.apcdcl.devptr,
        aps.aptid.devptr,
        aps.apbid.devptr,
        aps.aprank.devptr,
        aps.apff.devptr,
        turbines.foundx.devptr,
        turbines.hubx.devptr,
        turbines.pitch.devptr,
        turbines.pitchrate.devptr,
        turbines.tiprate.devptr,
        u.devptr,
        x.devptr,
        y.devptr,
        z.devptr,
        turbines.r/turbines.nappb,
        cpm.pdm_list[cpm.rank].shape,
        cpm.rank
    );
}

void FalmAlmDevCall::updateForce(
    APArray &aps,
    Matrix<Real> &x,
    Matrix<Real> &ff,
    Real euler_eps,
    CPM &cpm,
    dim3 block_dim
) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_distributeForce<<<grid_dim, block_dim>>>(
        aps.apx.devptr,
        aps.apff.devptr,
        x.devptr,
        ff.devptr,
        euler_eps,
        map.shape,
        map.offset
    );
}

}
