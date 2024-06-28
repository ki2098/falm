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

__host__ __device__ inline REAL trilinear_interpolate(
    REAL xp, REAL yp, REAL zp,
    REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5, REAL a6, REAL a7
) {
    REAL a8  = (1 - xp)*a0  + xp*a1;
    REAL a9  = (1 - xp)*a2  + xp*a3;
    REAL a10 = (1 - xp)*a4  + xp*a5;
    REAL a11 = (1 - xp)*a6  + xp*a7;
    REAL a12 = (1 - yp)*a8  + yp*a9;
    REAL a13 = (1 - yp)*a10 + yp*a11;
    return (1 - zp)*a12 + zp*a13;
}

__global__ void kernel_updateApx(
    MatrixFrame<REAL> *vapx,
    MatrixFrame<INT>  *vapi,
    MatrixFrame<REAL> *vapr,
    MatrixFrame<REAL> *vaptheta,
    MatrixFrame<int>  *vaptid,
    MatrixFrame<int>  *vapbid,
    MatrixFrame<int>  *vaprank,
    MatrixFrame<REAL> *vfoundx,
    MatrixFrame<REAL> *vhubx,
    MatrixFrame<REAL> *vpitch,
    MatrixFrame<REAL> *vtiprate,
    MatrixFrame<REAL> *vx,
    MatrixFrame<REAL> *vy,
    MatrixFrame<REAL> *vz,
    MatrixFrame<INT>  *vioffset,
    MatrixFrame<INT>  *vjoffset,
    MatrixFrame<INT>  *vkoffset,
    int nbpt,
    int nappb,
    REAL tt,
    INT3 mpi_shape
) {
    MatrixFrame<REAL> &apx = *vapx;
    MatrixFrame<INT> &api = *vapi;
    MatrixFrame<REAL> &apr = *vapr;
    MatrixFrame<REAL> &aptheta = *vaptheta;
    MatrixFrame<int>  &aptid = *vaptid;
    MatrixFrame<int>  &apbid = *vapbid;
    MatrixFrame<int>  &aprank = *vaprank;
    MatrixFrame<REAL> &foundx = *vfoundx;
    MatrixFrame<REAL> &hubx = *vhubx;
    MatrixFrame<REAL> &pitch = *vpitch;
    MatrixFrame<REAL> &tiprate = *vtiprate;
    MatrixFrame<REAL> &x = *vx, &y = *vy, &z = *vz;
    MatrixFrame<INT> &ioffset = *vioffset, &joffset = *vjoffset, &koffset = *vkoffset;
    int apid = GLOBAL_THREAD_IDX();
    if (apid < apx.shape[0]) {
        int tid = aptid(apid);
        REAL initial_theta = apbid(apid)*2.*Pi/nbpt + 0.5*Pi;
        REAL theta = initial_theta + tiprate(tid)*tt;
        aptheta(apid) = theta;

        REAL rt = apr(apid);
        REAL3 apvt = hubx(tid) + REAL3{{0., rt*cos(theta), rt*sin(theta)}};
        REAL3 apv  = pitch_only_vector_turbine2abs(apvt, pitch(tid));
        REAL apX = foundx(tid, 0) + apv[0];
        REAL apY = foundx(tid, 1) + apv[1];
        REAL apZ = foundx(tid, 2) + apv[2];
        INT  apI = find_interval(x.ptr, x.size, apX);
        INT  apJ = find_interval(y.ptr, y.size, apY);
        INT  apK = find_interval(z.ptr, z.size, apZ);

        apx(apid, 0) = apX;
        apx(apid, 1) = apY;
        apx(apid, 2) = apZ;
        api(apid, 0) = apI;
        api(apid, 1) = apJ;
        api(apid, 2) = apK;

        INT rankI = find_interval(ioffset.ptr, ioffset.size, apI);
        INT rankJ = find_interval(joffset.ptr, joffset.size, apJ);
        INT rankK = find_interval(koffset.ptr, koffset.size, apK);
        aprank(apid) = IDX(rankI, rankJ, rankK, mpi_shape);
    }
}

__global__ void kernel_calcApForce(
    MatrixFrame<REAL> *vapx,
    MatrixFrame<INT>  *vapi,
    MatrixFrame<REAL> *vapr,
    MatrixFrame<REAL> *vaptheta,
    MatrixFrame<REAL> *vapchord,
    MatrixFrame<REAL> *vaptwist,
    MatrixFrame<REAL> *vapcdcl,
    MatrixFrame<int>  *vaptid,
    MatrixFrame<int>  *vapbid,
    MatrixFrame<int>  *vaprank,
    MatrixFrame<REAL> *vapff,
    MatrixFrame<REAL> *vfoundx,
    MatrixFrame<REAL> *vhubx,
    MatrixFrame<REAL> *vpitch,
    MatrixFrame<REAL> *vpitchrate,
    MatrixFrame<REAL> *vtiprate,
    MatrixFrame<REAL> *vu,
    MatrixFrame<REAL> *vx,
    MatrixFrame<REAL> *vy,
    MatrixFrame<REAL> *vz,
    REAL dr,
    INT3 shape,
    int rank
) {
    MatrixFrame<REAL> &apx = *vapx;
    MatrixFrame<INT> &api = *vapi;
    MatrixFrame<REAL> &apr = *vapr;
    MatrixFrame<REAL> &aptheta = *vaptheta;
    MatrixFrame<REAL> &apchord = *vapchord;
    MatrixFrame<REAL> &aptwist = *vaptwist;
    MatrixFrame<REAL> &apcdcl = *vapcdcl;
    MatrixFrame<int>  &aptid = *vaptid;
    MatrixFrame<int>  &apbid = *vapbid;
    MatrixFrame<int>  &aprank = *vaprank;
    MatrixFrame<REAL> &apff = *vapff;
    MatrixFrame<REAL> &foundx = *vfoundx;
    MatrixFrame<REAL> &hubx = *vhubx;
    MatrixFrame<REAL> &pitch = *vpitch;
    MatrixFrame<REAL> &pitchrate = *vpitchrate;
    MatrixFrame<REAL> &tiprate = *vtiprate;
    MatrixFrame<REAL> &u = *vu, &x = *vx, &y = *vy, &z = *vz;

    INT apid = GLOBAL_THREAD_IDX();
    if (apid < apx.shape[0]) {
        if (aprank(apid) == rank) {
            int tid = aptid(apid);
            REAL tpitch = pitch(tid);
            REAL tpitchrate = pitchrate(tid);
            INT  apI = api(apid,0), apJ = api(apid,1), apK = api(apid,2);
            REAL apX = apx(apid,0), apY = apx(apid,1), apZ = apx(apid,2);
            REAL x0 = x(apI), x1 = x(apI+1);
            REAL y0 = y(apJ), y1 = y(apJ+1);
            REAL z0 = z(apK), z1 = z(apK+1);
            REAL xp = (apX - x0)/(x1 - x0);
            REAL yp = (apY - y0)/(y1 - y0);
            REAL zp = (apZ - z0)/(z1 - z0);
            INT idx0 = IDX(apI  ,apJ  ,apK  ,shape);
            INT idx1 = IDX(apI+1,apJ  ,apK  ,shape);
            INT idx2 = IDX(apI  ,apJ+1,apK  ,shape);
            INT idx3 = IDX(apI+1,apJ+1,apK  ,shape);
            INT idx4 = IDX(apI  ,apJ  ,apK+1,shape);
            INT idx5 = IDX(apI+1,apJ  ,apK+1,shape);
            INT idx6 = IDX(apI  ,apJ+1,apK+1,shape);
            INT idx7 = IDX(apI+1,apJ+1,apK+1,shape);
            REAL u0 = u(idx0, 0);
            REAL u1 = u(idx1, 0);
            REAL u2 = u(idx2, 0);
            REAL u3 = u(idx3, 0);
            REAL u4 = u(idx4, 0);
            REAL u5 = u(idx5, 0);
            REAL u6 = u(idx6, 0);
            REAL u7 = u(idx7, 0);
            REAL v0 = u(idx0, 1);
            REAL v1 = u(idx1, 1);
            REAL v2 = u(idx2, 1);
            REAL v3 = u(idx3, 1);
            REAL v4 = u(idx4, 1);
            REAL v5 = u(idx5, 1);
            REAL v6 = u(idx6, 1);
            REAL v7 = u(idx7, 1);
            REAL w0 = u(idx0, 2);
            REAL w1 = u(idx1, 2);
            REAL w2 = u(idx2, 2);
            REAL w3 = u(idx3, 2);
            REAL w4 = u(idx4, 2);
            REAL w5 = u(idx5, 2);
            REAL w6 = u(idx6, 2);
            REAL w7 = u(idx7, 2);
            REAL uc = trilinear_interpolate(xp, yp, zp, u0, u1, u2, u3, u4, u5, u6, u7);
            REAL vc = trilinear_interpolate(xp, yp, zp, v0, v1, v2, v3, v4, v5, v6, v7);
            REAL wc = trilinear_interpolate(xp, yp, zp, w0, w1, w2, w3, w4, w5, w6, w7);

            REAL3 X0{{foundx(tid,0), foundx(tid,1), foundx(tid,2)}};
            REAL3 X{{apX, apY, apZ}};
            REAL3 U{{uc, vc, wc}};
            REAL3 UAP = pitch_only_velocity_abs2turbine(U, X, X0, tpitch, tpitchrate);

            REAL theta = aptheta(apid);
            REAL ttiprate = tiprate(tid);
            REAL utt = sign(ttiprate)*(apr(apid)*ttiprate + UAP[1]*sin(theta) - UAP[2]*cos(theta));
            REAL uxt = UAP[0];
            REAL urel2 = square(uxt) + square(utt);
            REAL phi = atan2(uxt, utt);
            REAL attack = phi*180./Pi - aptwist(apid);
            REAL chord = apchord(apid);
            REAL cd, cl;
            getcdcl(vapcdcl, apid, attack, cd, cl);
            REAL fl = .5*cl*urel2*chord*dr;
            REAL fd = .5*cd*urel2*chord*dr;
            REAL fx = fl*cos(phi) + fd*sin(phi);
            REAL ft = fl*sin(phi) - fd*cos(phi);
            ft *= sign(ttiprate);
            REAL3 ff = pitch_only_vector_turbine2abs(REAL3{{-fx,ft*sin(theta),-ft*cos(theta)}}, tpitch);
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
    MatrixFrame<REAL> *vapx,
    MatrixFrame<REAL> *vapff,
    MatrixFrame<REAL> *vx,
    MatrixFrame<REAL> *vff,
    REAL euler_eps,
    INT3 shape,
    INT3 offset
) {
    MatrixFrame<REAL> &apx=*vapx, &apff=*vapff, &x=*vx, &ff=*vff;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] && j < shape[1] && k < shape[2]) {
        i += offset[0];
        j += offset[1];
        k += offset[2];
        INT cid = IDX(i,j,k,shape);
        REAL cx = x(cid,0);
        REAL cy = x(cid,1);
        REAL cz = x(cid,2);
        REAL ffx = 0, ffy = 0, ffz = 0;
        REAL eta = 1./(cube(euler_eps)*cbrt(square(Pi)));
        for (int apid = 0; apid < apff.shape[0]; apid ++) {
            REAL ax = apx(apid, 0);
            REAL ay = apx(apid, 1);
            REAL az = apx(apid, 2);
            REAL rr2 = square(cx-ax) + square(cy-ay) + square(cz-az);

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
    Matrix<REAL> &x, Matrix<REAL> &y, Matrix<REAL> &z,
    Matrix<REAL> &u,
    REAL tt,
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
    Matrix<REAL> &x,
    Matrix<REAL> &ff,
    REAL euler_eps,
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
