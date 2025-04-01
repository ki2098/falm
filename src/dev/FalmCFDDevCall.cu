#include "../FalmCFDDevCall.h"
#include "../FalmCFDScheme.h"
#include "devutil.cuh"

#define SQ(x) ((x)*(x))
#define CB(x) ((x)*(x)*(x))

namespace Falm {

__global__ void kernel_Cartesian_CalcPseudoU(
    const MatrixFrame<Real> *vun,
    const MatrixFrame<Real> *vu,
    const MatrixFrame<Real> *vuu,
    const MatrixFrame<Real> *vua,
    const MatrixFrame<Real> *vnut,
    const MatrixFrame<Real> *vkx,
    const MatrixFrame<Real> *vg,
    const MatrixFrame<Real> *vja,
    const MatrixFrame<Real> *vff,
    Flag advtype,
    Real               ReI,
    Real               dt,
    Int3              pdm_shape,
    Int3              map_shap,
    Int3              map_offset
) {
    const MatrixFrame<Real> &un=*vun, &u=*vu, &uu=*vuu, &ua=*vua, &nut=*vnut, &kx=*vkx, &g=*vg, &ja=*vja, &ff=*vff;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idxcc = IDX(i  , j  , k  , pdm_shape);
        Int idxe1 = IDX(i+1, j  , k  , pdm_shape);
        Int idxw1 = IDX(i-1, j  , k  , pdm_shape);
        Int idxn1 = IDX(i  , j+1, k  , pdm_shape);
        Int idxs1 = IDX(i  , j-1, k  , pdm_shape);
        Int idxt1 = IDX(i  , j  , k+1, pdm_shape);
        Int idxb1 = IDX(i  , j  , k-1, pdm_shape);
        Int idxe2 = IDX(i+2, j  , k  , pdm_shape);
        Int idxw2 = IDX(i-2, j  , k  , pdm_shape);
        Int idxn2 = IDX(i  , j+2, k  , pdm_shape);
        Int idxs2 = IDX(i  , j-2, k  , pdm_shape);
        Int idxt2 = IDX(i  , j  , k+2, pdm_shape);
        Int idxb2 = IDX(i  , j  , k-2, pdm_shape);
        Int idxE  = idxcc;
        Int idxW  = idxw1;
        Int idxN  = idxcc;
        Int idxS  = idxs1;
        Int idxT  = idxcc;
        Int idxB  = idxb1;
        Real uc    = u(idxcc, 0);
        Real vc    = u(idxcc, 1);
        Real wc    = u(idxcc, 2);
        Real Uabs  = fabs(uc * kx(idxcc, 0));
        Real Vabs  = fabs(vc * kx(idxcc, 1));
        Real Wabs  = fabs(wc * kx(idxcc, 2));
        Real UE    = uu(idxE, 0);
        Real UW    = uu(idxW, 0);
        Real VN    = uu(idxN, 1);
        Real VS    = uu(idxS, 1);
        Real WT    = uu(idxT, 2);
        Real WB    = uu(idxB, 2);
        Real nutcc = nut(idxcc);
        Real nute1 = nut(idxe1);
        Real nutw1 = nut(idxw1);
        Real nutn1 = nut(idxn1);
        Real nuts1 = nut(idxs1);
        Real nutt1 = nut(idxt1);
        Real nutb1 = nut(idxb1);
        Real gxcc  = g(idxcc, 0);
        Real gxe1  = g(idxe1, 0);
        Real gxw1  = g(idxw1, 0);
        Real gycc  = g(idxcc, 1);
        Real gyn1  = g(idxn1, 1);
        Real gys1  = g(idxs1, 1);
        Real gzcc  = g(idxcc, 2);
        Real gzt1  = g(idxt1, 2);
        Real gzb1  = g(idxb1, 2);
        Real jacob = ja(idxcc);

        Int d;
        Real ucc;
        Real ue1, ue2, uw1, uw2;
        Real un1, un2, us1, us2;
        Real ut1, ut2, ub1, ub2;
        Real adv, vis;

        d = 0;
        ucc = uc;
        ue1 = u(idxe1, d);
        uw1 = u(idxw1, d);
        un1 = u(idxn1, d);
        us1 = u(idxs1, d);
        ut1 = u(idxt1, d);
        ub1 = u(idxb1, d);
        ue2 = u(idxe2, d);
        uw2 = u(idxw2, d);
        un2 = u(idxn2, d);
        us2 = u(idxs2, d);
        ut2 = u(idxt2, d);
        ub2 = u(idxb2, d);
        if (advtype == AdvectionSchemeType::Upwind3) {
            adv = Upwind3rd(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                Uabs, Vabs, Wabs,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else if (advtype == AdvectionSchemeType::QUICK) {
            adv = QUICK(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else if (advtype == AdvectionSchemeType::UTOPIA) {
            adv = UTOPIA(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                Uabs, Vabs, Wabs,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else if (advtype == AdvectionSchemeType::KK) {
            adv = KK(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                Uabs, Vabs, Wabs,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else {
            adv = Upwind1st(
                ucc, ue1, uw1, un1, us1, ut1, ub1,
                UE, UW, VN, VS, WT, WB, jacob
            );
        }
        vis = Diffusion(
            ReI,
            ucc, ue1, uw1, un1, us1, ut1, ub1,
            nutcc, nute1, nutw1, nutn1, nuts1, nutt1, nutb1,
            gxcc, gxe1, gxw1,
            gycc, gyn1, gys1,
            gzcc, gzt1, gzb1,
            jacob
        );
        ua(idxcc, d) = un(idxcc, d) + dt * (- adv + vis + ff(idxcc, d));

        d = 1;
        ucc = vc;
        ue1 = u(idxe1, d);
        uw1 = u(idxw1, d);
        un1 = u(idxn1, d);
        us1 = u(idxs1, d);
        ut1 = u(idxt1, d);
        ub1 = u(idxb1, d);
        ue2 = u(idxe2, d);
        uw2 = u(idxw2, d);
        un2 = u(idxn2, d);
        us2 = u(idxs2, d);
        ut2 = u(idxt2, d);
        ub2 = u(idxb2, d);
        if (advtype == AdvectionSchemeType::Upwind3) {
            adv = Upwind3rd(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                Uabs, Vabs, Wabs,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else if (advtype == AdvectionSchemeType::QUICK) {
            adv = QUICK(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else if (advtype == AdvectionSchemeType::UTOPIA) {
            adv = UTOPIA(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                Uabs, Vabs, Wabs,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else if (advtype == AdvectionSchemeType::KK) {
            adv = KK(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                Uabs, Vabs, Wabs,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else {
            adv = Upwind1st(
                ucc, ue1, uw1, un1, us1, ut1, ub1,
                UE, UW, VN, VS, WT, WB, jacob
            );
        }
        vis = Diffusion(
            ReI,
            ucc, ue1, uw1, un1, us1, ut1, ub1,
            nutcc, nute1, nutw1, nutn1, nuts1, nutt1, nutb1,
            gxcc, gxe1, gxw1,
            gycc, gyn1, gys1,
            gzcc, gzt1, gzb1,
            jacob
        );
        ua(idxcc, d) = un(idxcc, d) + dt * (- adv + vis + ff(idxcc, d));

        d = 2;
        ucc = wc;
        ue1 = u(idxe1, d);
        uw1 = u(idxw1, d);
        un1 = u(idxn1, d);
        us1 = u(idxs1, d);
        ut1 = u(idxt1, d);
        ub1 = u(idxb1, d);
        ue2 = u(idxe2, d);
        uw2 = u(idxw2, d);
        un2 = u(idxn2, d);
        us2 = u(idxs2, d);
        ut2 = u(idxt2, d);
        ub2 = u(idxb2, d);
        if (advtype == AdvectionSchemeType::Upwind3) {
            adv = Upwind3rd(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                Uabs, Vabs, Wabs,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else if (advtype == AdvectionSchemeType::QUICK) {
            adv = QUICK(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else if (advtype == AdvectionSchemeType::UTOPIA) {
            adv = UTOPIA(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                Uabs, Vabs, Wabs,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else if (advtype == AdvectionSchemeType::KK) {
            adv = KK(
                ucc,
                ue1, ue2, uw1, uw2,
                un1, un2, us1, us2,
                ut1, ut2, ub1, ub2,
                Uabs, Vabs, Wabs,
                UE, UW, VN, VS, WT, WB,
                jacob
            );
        } else {
            adv = Upwind1st(
                ucc, ue1, uw1, un1, us1, ut1, ub1,
                UE, UW, VN, VS, WT, WB, jacob
            );
        }
        vis = Diffusion(
            ReI,
            ucc, ue1, uw1, un1, us1, ut1, ub1,
            nutcc, nute1, nutw1, nutn1, nuts1, nutt1, nutb1,
            gxcc, gxe1, gxw1,
            gycc, gyn1, gys1,
            gzcc, gzt1, gzb1,
            jacob
        );
        ua(idxcc, d) = un(idxcc, d) + dt * (- adv + vis + ff(idxcc, d));
    }
    // pdm_shape += {1, 1, 1};
}

__global__ void kernel_Cartesian_UtoCU (
    const MatrixFrame<Real> *vu,
    const MatrixFrame<Real> *vuc,
    const MatrixFrame<Real> *vkx,
    const MatrixFrame<Real> *vja,
    Int3              pdm_shape,
    Int3              map_shap,
    Int3              map_offset
) {
    const MatrixFrame<Real> &u=*vu, &uc=*vuc, &kx=*vkx, &ja=*vja;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idx = IDX(i, j, k, pdm_shape);
        Real jacob = ja(idx);
        uc(idx, 0) = jacob * kx(idx, 0) * u(idx, 0);
        uc(idx, 1) = jacob * kx(idx, 1) * u(idx, 1);
        uc(idx, 2) = jacob * kx(idx, 2) * u(idx, 2);
    }
}

__global__ void kernel_Cartesian_InterpolateCU(
    const MatrixFrame<Real> *vuu,
    const MatrixFrame<Real> *vuc,
    Int3              pdm_shape,
    Int3              map_shap,
    Int3              map_offset
) {
    const MatrixFrame<Real> &uu=*vuu, &uc=*vuc;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idxcc = IDX(i  , j  , k  , pdm_shape);
        Int idxe1 = IDX(i+1, j  , k  , pdm_shape);
        Int idxn1 = IDX(i  , j+1, k  , pdm_shape);
        Int idxt1 = IDX(i  , j  , k+1, pdm_shape);
        uu(idxcc, 0) = 0.5 * (uc(idxcc, 0) + uc(idxe1, 0));
        uu(idxcc, 1) = 0.5 * (uc(idxcc, 1) + uc(idxn1, 1));
        uu(idxcc, 2) = 0.5 * (uc(idxcc, 2) + uc(idxt1, 2));
    }
}

__global__ void kernel_Cartesian_ProjectPGrid(
    const MatrixFrame<Real> *vu,
    const MatrixFrame<Real> *vua,
    const MatrixFrame<Real> *vp,
    const MatrixFrame<Real> *vkx,
    Real               dt,
    Int3              pdm_shape,
    Int3              map_shap,
    Int3              map_offset
) {
    const MatrixFrame<Real> &u=*vu, &ua=*vua, &p=*vp, &kx=*vkx;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idxcc = IDX(i, j, k, pdm_shape);
        Real dpx = 0.5 * kx(idxcc, 0) * (p(IDX(i+1, j  , k  , pdm_shape)) - p(IDX(i-1, j  , k  , pdm_shape)));
        Real dpy = 0.5 * kx(idxcc, 1) * (p(IDX(i  , j+1, k  , pdm_shape)) - p(IDX(i  , j-1, k  , pdm_shape)));
        Real dpz = 0.5 * kx(idxcc, 2) * (p(IDX(i  , j  , k+1, pdm_shape)) - p(IDX(i  , j  , k-1, pdm_shape)));
        u(idxcc, 0) = ua(idxcc, 0) - dt * dpx;
        u(idxcc, 1) = ua(idxcc, 1) - dt * dpy;
        u(idxcc, 2) = ua(idxcc, 2) - dt * dpz;
    }
}

__global__ void kernel_Cartesian_ProjectPFace(
    const MatrixFrame<Real> *vuu,
    const MatrixFrame<Real> *vuua,
    const MatrixFrame<Real> *vp,
    const MatrixFrame<Real> *vg,
    Real               dt,
    Int3              pdm_shape,
    Int3              map_shap,
    Int3              map_offset
) {
    const MatrixFrame<Real> &uu=*vuu, &uua=*vuua, &p=*vp, &g=*vg;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idxcc = IDX(i  , j  , k  , pdm_shape);
        Int idxe1 = IDX(i+1, j  , k  , pdm_shape);
        Int idxn1 = IDX(i  , j+1, k  , pdm_shape);
        Int idxt1 = IDX(i  , j  , k+1, pdm_shape);
        Real pcc = p(idxcc);
        Real dpx = 0.5 * (g(idxcc, 0) + g(idxe1, 0)) * (p(idxe1) - pcc);
        Real dpy = 0.5 * (g(idxcc, 1) + g(idxn1, 1)) * (p(idxn1) - pcc);
        Real dpz = 0.5 * (g(idxcc, 2) + g(idxt1, 2)) * (p(idxt1) - pcc);
        uu(idxcc, 0) = uua(idxcc, 0) - dt * dpx;
        uu(idxcc, 1) = uua(idxcc, 1) - dt * dpy;
        uu(idxcc, 2) = uua(idxcc, 2) - dt * dpz;
    }
}

__global__ void kernel_Cartesian_Smagorinsky(
    const MatrixFrame<Real> *vu,
    const MatrixFrame<Real> *vnut,
    const MatrixFrame<Real> *vx,
    const MatrixFrame<Real> *vkx,
    const MatrixFrame<Real> *vja,
    Real               Cs,
    Int3              pdm_shape,
    Int3              map_shap,
    Int3              map_offset            
) {
    const MatrixFrame<Real> &u=*vu, &nut=*vnut, &x=*vx, &kx=*vkx, &ja=*vja;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idxcc = IDX(i  , j  , k  , pdm_shape);
        Int idxe1 = IDX(i+1, j  , k  , pdm_shape);
        Int idxw1 = IDX(i-1, j  , k  , pdm_shape);
        Int idxn1 = IDX(i  , j+1, k  , pdm_shape);
        Int idxs1 = IDX(i  , j-1, k  , pdm_shape);
        Int idxt1 = IDX(i  , j  , k+1, pdm_shape);
        Int idxb1 = IDX(i  , j  , k-1, pdm_shape);
        Real kxx = kx(idxcc, 0);
        Real kxy = kx(idxcc, 1);
        Real kxz = kx(idxcc, 2);
        Real ue1, uw1, un1, us1, ut1, ub1;
        ue1 = u(idxe1, 0);
        uw1 = u(idxw1, 0);
        un1 = u(idxn1, 0);
        us1 = u(idxs1, 0);
        ut1 = u(idxt1, 0);
        ub1 = u(idxb1, 0);
        Real dux = 0.5 * kxx * (ue1 - uw1);
        Real duy = 0.5 * kxy * (un1 - us1);
        Real duz = 0.5 * kxz * (ut1 - ub1);
        ue1 = u(idxe1, 1);
        uw1 = u(idxw1, 1);
        un1 = u(idxn1, 1);
        us1 = u(idxs1, 1);
        ut1 = u(idxt1, 1);
        ub1 = u(idxb1, 1);
        Real dvx = 0.5 * kxx * (ue1 - uw1);
        Real dvy = 0.5 * kxy * (un1 - us1);
        Real dvz = 0.5 * kxz * (ut1 - ub1);
        ue1 = u(idxe1, 2);
        uw1 = u(idxw1, 2);
        un1 = u(idxn1, 2);
        us1 = u(idxs1, 2);
        ut1 = u(idxt1, 2);
        ub1 = u(idxb1, 2);
        Real dwx = 0.5 * kxx * (ue1 - uw1);
        Real dwy = 0.5 * kxy * (un1 - us1);
        Real dwz = 0.5 * kxz * (ut1 - ub1);
        Real d1 = 2 * SQ(dux);
        Real d2 = 2 * SQ(dvy);
        Real d3 = 2 * SQ(dwz);
        Real d4 = SQ(duy + dvx);
        Real d5 = SQ(dvz + dwy);
        Real d6 = SQ(duz + dwx);
        Real Du = sqrt(d1 + d2 + d3 + d4 + d5 + d6);
        Real De = cbrt(ja(idxcc));
        Real lc = Cs * De;
        nut(idxcc) = SQ(lc) * Du;
    }
}

__global__ void kernel_Cartesian_CSM(
    const MatrixFrame<Real> *vu,
    const MatrixFrame<Real> *vnut,
    const MatrixFrame<Real> *vx,
    const MatrixFrame<Real> *vkx,
    const MatrixFrame<Real> *vja,
    Int3              pdm_shape,
    Int3              map_shap,
    Int3              map_offset   
) {
    const MatrixFrame<Real> &u=*vu, &nut=*vnut, &x=*vx, &kx=*vkx, &ja=*vja;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idxcc = IDX(i  , j  , k  , pdm_shape);
        Int idxe1 = IDX(i+1, j  , k  , pdm_shape);
        Int idxw1 = IDX(i-1, j  , k  , pdm_shape);
        Int idxn1 = IDX(i  , j+1, k  , pdm_shape);
        Int idxs1 = IDX(i  , j-1, k  , pdm_shape);
        Int idxt1 = IDX(i  , j  , k+1, pdm_shape);
        Int idxb1 = IDX(i  , j  , k-1, pdm_shape);
        Real kxx = kx(idxcc, 0);
        Real kxy = kx(idxcc, 1);
        Real kxz = kx(idxcc, 2);
        Real ue1, uw1, un1, us1, ut1, ub1;
        ue1 = u(idxe1, 0);
        uw1 = u(idxw1, 0);
        un1 = u(idxn1, 0);
        us1 = u(idxs1, 0);
        ut1 = u(idxt1, 0);
        ub1 = u(idxb1, 0);
        Real dux = 0.5 * kxx * (ue1 - uw1);
        Real duy = 0.5 * kxy * (un1 - us1);
        Real duz = 0.5 * kxz * (ut1 - ub1);
        ue1 = u(idxe1, 1);
        uw1 = u(idxw1, 1);
        un1 = u(idxn1, 1);
        us1 = u(idxs1, 1);
        ut1 = u(idxt1, 1);
        ub1 = u(idxb1, 1);
        Real dvx = 0.5 * kxx * (ue1 - uw1);
        Real dvy = 0.5 * kxy * (un1 - us1);
        Real dvz = 0.5 * kxz * (ut1 - ub1);
        ue1 = u(idxe1, 2);
        uw1 = u(idxw1, 2);
        un1 = u(idxn1, 2);
        us1 = u(idxs1, 2);
        ut1 = u(idxt1, 2);
        ub1 = u(idxb1, 2);
        Real dwx = 0.5 * kxx * (ue1 - uw1);
        Real dwy = 0.5 * kxy * (un1 - us1);
        Real dwz = 0.5 * kxz * (ut1 - ub1);
        Real d1 = 2 * SQ(dux);
        Real d2 = 2 * SQ(dvy);
        Real d3 = 2 * SQ(dwz);
        Real d4 = SQ(duy + dvx);
        Real d5 = SQ(dvz + dwy);
        Real d6 = SQ(duz + dwx);
        Real Du = sqrt(d1 + d2 + d3 + d4 + d5 + d6);
        Real De = cbrt(ja(idxcc));
        Real e, q;
        e  = SQ(dux) + SQ(duy) + SQ(duz);
        e += SQ(dvx) + SQ(dvy) + SQ(dvz);
        e += SQ(dwx) + SQ(dwy) + SQ(dwz);
        e *= 0.5;
        q  = dux * dux + duy * dvx + duz * dwx;
        q += dvx * duy + dvy * dvy + dvz * dwy;
        q += dwx * duz + dwy * dvz + dwz * dwz;
        q *= - 0.5;
        Real  fcs = (q + copysign(1e-9, q)) / (e + copysign(1e-9, e));
        Real afcs = fabs(fcs);
        Real CCsm = sqrt(CB(fcs)) * (1 - fcs) / 22.0;
        nut(idxcc) = CCsm * SQ(De) * Du;
    }
}

__global__ void kernel_Cartesian_Divergence(
    const MatrixFrame<Real> *vuu,
    const MatrixFrame<Real> *vdiv,
    const MatrixFrame<Real> *vja,
    Int3              pdm_shap,
    Int3              map_shap,
    Int3              map_offset
) {
    const MatrixFrame<Real> &uu=*vuu, &div=*vdiv, &ja=*vja;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idxc = IDX(i, j, k, pdm_shap);
        Real UE = uu(idxc                    , 0);
        Real UW = uu(IDX(i-1, j, k, pdm_shap), 0);
        Real VN = uu(idxc                    , 1);
        Real VS = uu(IDX(i, j-1, k, pdm_shap), 1);
        Real WT = uu(idxc                    , 2);
        Real WB = uu(IDX(i, j, k-1, pdm_shap), 2);
        div(idxc) = (UE - UW + VN - VS + WT - WB) / ja(idxc);
    }
}

__global__ void kernel_Vortcity(
    const MatrixFrame<Real> *vu,
    const MatrixFrame<Real> *vkx,
    const MatrixFrame<Real> *vvrt,
    Int3              pdm_shap,
    Int3              map_shap,
    Int3              map_offset
) {
    const MatrixFrame<Real> &u = *vu, &kx = *vkx, &vrt = *vvrt;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int idxc = IDX(i, j, k, pdm_shap);
        Real kxx = kx(idxc, 0);
        Real kxy = kx(idxc, 1);
        Real kxz = kx(idxc, 2);
        Real dvdx = 0.5 * kxx * (u(IDX(i+1,j,k,pdm_shap),1) - u(IDX(i-1,j,k,pdm_shap),1));
        Real dwdx = 0.5 * kxx * (u(IDX(i+1,j,k,pdm_shap),2) - u(IDX(i-1,j,k,pdm_shap),2));
        Real dudy = 0.5 * kxy * (u(IDX(i,j+1,k,pdm_shap),0) - u(IDX(i,j-1,k,pdm_shap),0));
        Real dwdy = 0.5 * kxy * (u(IDX(i,j+1,k,pdm_shap),2) - u(IDX(i,j-1,k,pdm_shap),2));
        Real dudz = 0.5 * kxz * (u(IDX(i,j,k+1,pdm_shap),0) - u(IDX(i,j,k-1,pdm_shap),0));
        Real dvdz = 0.5 * kxz * (u(IDX(i,j,k+1,pdm_shap),1) - u(IDX(i,j,k-1,pdm_shap),1));
        vrt(idxc, 0) = dwdy - dvdz;
        vrt(idxc, 1) = dudz - dwdx;
        vrt(idxc, 2) = dvdx - dudy;
    }
}

void FalmCFDDevCall::FSPseudoU(
    Matrix<Real> &un,
    Matrix<Real> &u,
    Matrix<Real> &uu,
    Matrix<Real> &ua,
    Matrix<Real> &nut,
    Matrix<Real> &kx,
    Matrix<Real> &g,
    Matrix<Real> &ja,
    Matrix<Real> &ff,
    Real dt,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    Stream        stream
) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_Cartesian_CalcPseudoU<<<grid_dim, block_dim, 0, stream>>>(
        un.devptr,
        u.devptr,
        uu.devptr,
        ua.devptr,
        nut.devptr,
        kx.devptr,
        g.devptr,
        ja.devptr,
        ff.devptr,
        AdvScheme,
        ReI,
        dt,
        pdm.shape,
        map.shape,
        map.offset
    );
}

void FalmCFDDevCall::UtoCU(
    Matrix<Real> &u,
    Matrix<Real> &uc,
    Matrix<Real> &kx,
    Matrix<Real> &ja,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    Stream        stream
) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_Cartesian_UtoCU<<<grid_dim, block_dim, 0, stream>>>(
        u.devptr,
        uc.devptr,
        kx.devptr,
        ja.devptr,
        pdm.shape,
        map.shape,
        map.offset
    );
}

void FalmCFDDevCall::InterpolateCU(
    Matrix<Real> &uu,
    Matrix<Real> &uc,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    Stream        stream
) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_Cartesian_InterpolateCU<<<grid_dim, block_dim, 0, stream>>>(
        uu.devptr,
        uc.devptr,
        pdm.shape,
        map.shape,
        map.offset
    );
}

void FalmCFDDevCall::ProjectPGrid(
    Matrix<Real> &u,
    Matrix<Real> &ua,
    Matrix<Real> &p,
    Matrix<Real> &kx,
    Real dt,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    Stream        stream
) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_Cartesian_ProjectPGrid<<<grid_dim, block_dim, 0, stream>>>(
        u.devptr,
        ua.devptr,
        p.devptr,
        kx.devptr,
        dt,
        pdm.shape,
        map.shape,
        map.offset
    );
}

void FalmCFDDevCall::ProjectPFace(
    Matrix<Real> &uu,
    Matrix<Real> &uua,
    Matrix<Real> &p,
    Matrix<Real> &g,
    Real dt,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    Stream        stream
) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_Cartesian_ProjectPFace<<<grid_dim, block_dim, 0, stream>>>(
        uu.devptr,
        uua.devptr,
        p.devptr,
        g.devptr,
        dt,
        pdm.shape,
        map.shape,
        map.offset
    );
}

void FalmCFDDevCall::SGS(
    Matrix<Real> &u,
    Matrix<Real> &nut,
    Matrix<Real> &x,
    Matrix<Real> &kx,
    Matrix<Real> &ja,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    Stream        stream
) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    if (SGSModel == SGSType::Smagorinsky) {
        kernel_Cartesian_Smagorinsky<<<grid_dim, block_dim, 0, stream>>>(
            u.devptr,
            nut.devptr,
            x.devptr,
            kx.devptr,
            ja.devptr,
            CSmagorinsky,
            pdm.shape,
            map.shape,
            map.offset
        );
    } else if (SGSModel == SGSType::CSM) {
        kernel_Cartesian_CSM<<<grid_dim, block_dim, 0, stream>>>(
            u.devptr,
            nut.devptr,
            x.devptr,
            kx.devptr,
            ja.devptr,
            pdm.shape,
            map.shape,
            map.offset
        );
    }
}

void FalmCFDDevCall::Divergence(
    Matrix<Real> &uu,
    Matrix<Real> &div,
    Matrix<Real> &ja,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    Stream        stream
) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_Cartesian_Divergence<<<grid_dim, block_dim, 0, stream>>>(
        uu.devptr,
        div.devptr,
        ja.devptr,
        pdm.shape,
        map.shape,
        map.offset
    );
}

void FalmCFDDevCall::Vortcity(
    Matrix<Real> &u,
    Matrix<Real> &kx,
    Matrix<Real> &vrt,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    Stream        stream
) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_Vortcity<<<grid_dim, block_dim, 0, stream>>>(
        u.devptr, kx.devptr, vrt.devptr,
        pdm.shape, map.shape, map.offset
    );
}

}
