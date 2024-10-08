#include "../FalmCFDDevCall.h"
#include "../FalmCFDScheme.h"
#include "devutil.cuh"

#define SQ(x) ((x)*(x))
#define CB(x) ((x)*(x)*(x))

namespace Falm {

__global__ void kernel_Cartesian_CalcPseudoU(
    const MatrixFrame<REAL> *vun,
    const MatrixFrame<REAL> *vu,
    const MatrixFrame<REAL> *vuu,
    const MatrixFrame<REAL> *vua,
    const MatrixFrame<REAL> *vnut,
    const MatrixFrame<REAL> *vkx,
    const MatrixFrame<REAL> *vg,
    const MatrixFrame<REAL> *vja,
    const MatrixFrame<REAL> *vff,
    FLAG advtype,
    REAL               ReI,
    REAL               dt,
    INT3              pdm_shape,
    INT3              map_shap,
    INT3              map_offset
) {
    const MatrixFrame<REAL> &un=*vun, &u=*vu, &uu=*vuu, &ua=*vua, &nut=*vnut, &kx=*vkx, &g=*vg, &ja=*vja, &ff=*vff;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxcc = IDX(i  , j  , k  , pdm_shape);
        INT idxe1 = IDX(i+1, j  , k  , pdm_shape);
        INT idxw1 = IDX(i-1, j  , k  , pdm_shape);
        INT idxn1 = IDX(i  , j+1, k  , pdm_shape);
        INT idxs1 = IDX(i  , j-1, k  , pdm_shape);
        INT idxt1 = IDX(i  , j  , k+1, pdm_shape);
        INT idxb1 = IDX(i  , j  , k-1, pdm_shape);
        INT idxe2 = IDX(i+2, j  , k  , pdm_shape);
        INT idxw2 = IDX(i-2, j  , k  , pdm_shape);
        INT idxn2 = IDX(i  , j+2, k  , pdm_shape);
        INT idxs2 = IDX(i  , j-2, k  , pdm_shape);
        INT idxt2 = IDX(i  , j  , k+2, pdm_shape);
        INT idxb2 = IDX(i  , j  , k-2, pdm_shape);
        INT idxE  = idxcc;
        INT idxW  = idxw1;
        INT idxN  = idxcc;
        INT idxS  = idxs1;
        INT idxT  = idxcc;
        INT idxB  = idxb1;
        REAL uc    = u(idxcc, 0);
        REAL vc    = u(idxcc, 1);
        REAL wc    = u(idxcc, 2);
        REAL Uabs  = fabs(uc * kx(idxcc, 0));
        REAL Vabs  = fabs(vc * kx(idxcc, 1));
        REAL Wabs  = fabs(wc * kx(idxcc, 2));
        REAL UE    = uu(idxE, 0);
        REAL UW    = uu(idxW, 0);
        REAL VN    = uu(idxN, 1);
        REAL VS    = uu(idxS, 1);
        REAL WT    = uu(idxT, 2);
        REAL WB    = uu(idxB, 2);
        REAL nutcc = nut(idxcc);
        REAL nute1 = nut(idxe1);
        REAL nutw1 = nut(idxw1);
        REAL nutn1 = nut(idxn1);
        REAL nuts1 = nut(idxs1);
        REAL nutt1 = nut(idxt1);
        REAL nutb1 = nut(idxb1);
        REAL gxcc  = g(idxcc, 0);
        REAL gxe1  = g(idxe1, 0);
        REAL gxw1  = g(idxw1, 0);
        REAL gycc  = g(idxcc, 1);
        REAL gyn1  = g(idxn1, 1);
        REAL gys1  = g(idxs1, 1);
        REAL gzcc  = g(idxcc, 2);
        REAL gzt1  = g(idxt1, 2);
        REAL gzb1  = g(idxb1, 2);
        REAL jacob = ja(idxcc);

        INT d;
        REAL ucc;
        REAL ue1, ue2, uw1, uw2;
        REAL un1, un2, us1, us2;
        REAL ut1, ut2, ub1, ub2;
        REAL adv, vis;

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
    const MatrixFrame<REAL> *vu,
    const MatrixFrame<REAL> *vuc,
    const MatrixFrame<REAL> *vkx,
    const MatrixFrame<REAL> *vja,
    INT3              pdm_shape,
    INT3              map_shap,
    INT3              map_offset
) {
    const MatrixFrame<REAL> &u=*vu, &uc=*vuc, &kx=*vkx, &ja=*vja;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        REAL jacob = ja(idx);
        uc(idx, 0) = jacob * kx(idx, 0) * u(idx, 0);
        uc(idx, 1) = jacob * kx(idx, 1) * u(idx, 1);
        uc(idx, 2) = jacob * kx(idx, 2) * u(idx, 2);
    }
}

__global__ void kernel_Cartesian_InterpolateCU(
    const MatrixFrame<REAL> *vuu,
    const MatrixFrame<REAL> *vuc,
    INT3              pdm_shape,
    INT3              map_shap,
    INT3              map_offset
) {
    const MatrixFrame<REAL> &uu=*vuu, &uc=*vuc;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxcc = IDX(i  , j  , k  , pdm_shape);
        INT idxe1 = IDX(i+1, j  , k  , pdm_shape);
        INT idxn1 = IDX(i  , j+1, k  , pdm_shape);
        INT idxt1 = IDX(i  , j  , k+1, pdm_shape);
        uu(idxcc, 0) = 0.5 * (uc(idxcc, 0) + uc(idxe1, 0));
        uu(idxcc, 1) = 0.5 * (uc(idxcc, 1) + uc(idxn1, 1));
        uu(idxcc, 2) = 0.5 * (uc(idxcc, 2) + uc(idxt1, 2));
    }
}

__global__ void kernel_Cartesian_ProjectPGrid(
    const MatrixFrame<REAL> *vu,
    const MatrixFrame<REAL> *vua,
    const MatrixFrame<REAL> *vp,
    const MatrixFrame<REAL> *vkx,
    REAL               dt,
    INT3              pdm_shape,
    INT3              map_shap,
    INT3              map_offset
) {
    const MatrixFrame<REAL> &u=*vu, &ua=*vua, &p=*vp, &kx=*vkx;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxcc = IDX(i, j, k, pdm_shape);
        REAL dpx = 0.5 * kx(idxcc, 0) * (p(IDX(i+1, j  , k  , pdm_shape)) - p(IDX(i-1, j  , k  , pdm_shape)));
        REAL dpy = 0.5 * kx(idxcc, 1) * (p(IDX(i  , j+1, k  , pdm_shape)) - p(IDX(i  , j-1, k  , pdm_shape)));
        REAL dpz = 0.5 * kx(idxcc, 2) * (p(IDX(i  , j  , k+1, pdm_shape)) - p(IDX(i  , j  , k-1, pdm_shape)));
        u(idxcc, 0) = ua(idxcc, 0) - dt * dpx;
        u(idxcc, 1) = ua(idxcc, 1) - dt * dpy;
        u(idxcc, 2) = ua(idxcc, 2) - dt * dpz;
    }
}

__global__ void kernel_Cartesian_ProjectPFace(
    const MatrixFrame<REAL> *vuu,
    const MatrixFrame<REAL> *vuua,
    const MatrixFrame<REAL> *vp,
    const MatrixFrame<REAL> *vg,
    REAL               dt,
    INT3              pdm_shape,
    INT3              map_shap,
    INT3              map_offset
) {
    const MatrixFrame<REAL> &uu=*vuu, &uua=*vuua, &p=*vp, &g=*vg;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxcc = IDX(i  , j  , k  , pdm_shape);
        INT idxe1 = IDX(i+1, j  , k  , pdm_shape);
        INT idxn1 = IDX(i  , j+1, k  , pdm_shape);
        INT idxt1 = IDX(i  , j  , k+1, pdm_shape);
        REAL pcc = p(idxcc);
        REAL dpx = 0.5 * (g(idxcc, 0) + g(idxe1, 0)) * (p(idxe1) - pcc);
        REAL dpy = 0.5 * (g(idxcc, 1) + g(idxn1, 1)) * (p(idxn1) - pcc);
        REAL dpz = 0.5 * (g(idxcc, 2) + g(idxt1, 2)) * (p(idxt1) - pcc);
        uu(idxcc, 0) = uua(idxcc, 0) - dt * dpx;
        uu(idxcc, 1) = uua(idxcc, 1) - dt * dpy;
        uu(idxcc, 2) = uua(idxcc, 2) - dt * dpz;
    }
}

__global__ void kernel_Cartesian_Smagorinsky(
    const MatrixFrame<REAL> *vu,
    const MatrixFrame<REAL> *vnut,
    const MatrixFrame<REAL> *vx,
    const MatrixFrame<REAL> *vkx,
    const MatrixFrame<REAL> *vja,
    REAL               Cs,
    INT3              pdm_shape,
    INT3              map_shap,
    INT3              map_offset            
) {
    const MatrixFrame<REAL> &u=*vu, &nut=*vnut, &x=*vx, &kx=*vkx, &ja=*vja;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxcc = IDX(i  , j  , k  , pdm_shape);
        INT idxe1 = IDX(i+1, j  , k  , pdm_shape);
        INT idxw1 = IDX(i-1, j  , k  , pdm_shape);
        INT idxn1 = IDX(i  , j+1, k  , pdm_shape);
        INT idxs1 = IDX(i  , j-1, k  , pdm_shape);
        INT idxt1 = IDX(i  , j  , k+1, pdm_shape);
        INT idxb1 = IDX(i  , j  , k-1, pdm_shape);
        REAL kxx = kx(idxcc, 0);
        REAL kxy = kx(idxcc, 1);
        REAL kxz = kx(idxcc, 2);
        REAL ue1, uw1, un1, us1, ut1, ub1;
        ue1 = u(idxe1, 0);
        uw1 = u(idxw1, 0);
        un1 = u(idxn1, 0);
        us1 = u(idxs1, 0);
        ut1 = u(idxt1, 0);
        ub1 = u(idxb1, 0);
        REAL dux = 0.5 * kxx * (ue1 - uw1);
        REAL duy = 0.5 * kxy * (un1 - us1);
        REAL duz = 0.5 * kxz * (ut1 - ub1);
        ue1 = u(idxe1, 1);
        uw1 = u(idxw1, 1);
        un1 = u(idxn1, 1);
        us1 = u(idxs1, 1);
        ut1 = u(idxt1, 1);
        ub1 = u(idxb1, 1);
        REAL dvx = 0.5 * kxx * (ue1 - uw1);
        REAL dvy = 0.5 * kxy * (un1 - us1);
        REAL dvz = 0.5 * kxz * (ut1 - ub1);
        ue1 = u(idxe1, 2);
        uw1 = u(idxw1, 2);
        un1 = u(idxn1, 2);
        us1 = u(idxs1, 2);
        ut1 = u(idxt1, 2);
        ub1 = u(idxb1, 2);
        REAL dwx = 0.5 * kxx * (ue1 - uw1);
        REAL dwy = 0.5 * kxy * (un1 - us1);
        REAL dwz = 0.5 * kxz * (ut1 - ub1);
        REAL d1 = 2 * SQ(dux);
        REAL d2 = 2 * SQ(dvy);
        REAL d3 = 2 * SQ(dwz);
        REAL d4 = SQ(duy + dvx);
        REAL d5 = SQ(dvz + dwy);
        REAL d6 = SQ(duz + dwx);
        REAL Du = sqrt(d1 + d2 + d3 + d4 + d5 + d6);
        REAL De = cbrt(ja(idxcc));
        REAL lc = Cs * De;
        nut(idxcc) = SQ(lc) * Du;
    }
}

__global__ void kernel_Cartesian_CSM(
    const MatrixFrame<REAL> *vu,
    const MatrixFrame<REAL> *vnut,
    const MatrixFrame<REAL> *vx,
    const MatrixFrame<REAL> *vkx,
    const MatrixFrame<REAL> *vja,
    INT3              pdm_shape,
    INT3              map_shap,
    INT3              map_offset   
) {
    const MatrixFrame<REAL> &u=*vu, &nut=*vnut, &x=*vx, &kx=*vkx, &ja=*vja;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxcc = IDX(i  , j  , k  , pdm_shape);
        INT idxe1 = IDX(i+1, j  , k  , pdm_shape);
        INT idxw1 = IDX(i-1, j  , k  , pdm_shape);
        INT idxn1 = IDX(i  , j+1, k  , pdm_shape);
        INT idxs1 = IDX(i  , j-1, k  , pdm_shape);
        INT idxt1 = IDX(i  , j  , k+1, pdm_shape);
        INT idxb1 = IDX(i  , j  , k-1, pdm_shape);
        REAL kxx = kx(idxcc, 0);
        REAL kxy = kx(idxcc, 1);
        REAL kxz = kx(idxcc, 2);
        REAL ue1, uw1, un1, us1, ut1, ub1;
        ue1 = u(idxe1, 0);
        uw1 = u(idxw1, 0);
        un1 = u(idxn1, 0);
        us1 = u(idxs1, 0);
        ut1 = u(idxt1, 0);
        ub1 = u(idxb1, 0);
        REAL dux = 0.5 * kxx * (ue1 - uw1);
        REAL duy = 0.5 * kxy * (un1 - us1);
        REAL duz = 0.5 * kxz * (ut1 - ub1);
        ue1 = u(idxe1, 1);
        uw1 = u(idxw1, 1);
        un1 = u(idxn1, 1);
        us1 = u(idxs1, 1);
        ut1 = u(idxt1, 1);
        ub1 = u(idxb1, 1);
        REAL dvx = 0.5 * kxx * (ue1 - uw1);
        REAL dvy = 0.5 * kxy * (un1 - us1);
        REAL dvz = 0.5 * kxz * (ut1 - ub1);
        ue1 = u(idxe1, 2);
        uw1 = u(idxw1, 2);
        un1 = u(idxn1, 2);
        us1 = u(idxs1, 2);
        ut1 = u(idxt1, 2);
        ub1 = u(idxb1, 2);
        REAL dwx = 0.5 * kxx * (ue1 - uw1);
        REAL dwy = 0.5 * kxy * (un1 - us1);
        REAL dwz = 0.5 * kxz * (ut1 - ub1);
        REAL d1 = 2 * SQ(dux);
        REAL d2 = 2 * SQ(dvy);
        REAL d3 = 2 * SQ(dwz);
        REAL d4 = SQ(duy + dvx);
        REAL d5 = SQ(dvz + dwy);
        REAL d6 = SQ(duz + dwx);
        REAL Du = sqrt(d1 + d2 + d3 + d4 + d5 + d6);
        REAL De = cbrt(ja(idxcc));
        REAL e, q;
        e  = SQ(dux) + SQ(duy) + SQ(duz);
        e += SQ(dvx) + SQ(dvy) + SQ(dvz);
        e += SQ(dwx) + SQ(dwy) + SQ(dwz);
        e *= 0.5;
        q  = dux * dux + duy * dvx + duz * dwx;
        q += dvx * duy + dvy * dvy + dvz * dwy;
        q += dwx * duz + dwy * dvz + dwz * dwz;
        q *= - 0.5;
        REAL  fcs = (q + copysign(1e-9, q)) / (e + copysign(1e-9, e));
        REAL afcs = fabs(fcs);
        REAL CCsm = sqrt(CB(fcs)) * (1 - fcs) / 22.0;
        nut(idxcc) = CCsm * SQ(De) * Du;
    }
}

__global__ void kernel_Cartesian_Divergence(
    const MatrixFrame<REAL> *vuu,
    const MatrixFrame<REAL> *vdiv,
    const MatrixFrame<REAL> *vja,
    INT3              pdm_shap,
    INT3              map_shap,
    INT3              map_offset
) {
    const MatrixFrame<REAL> &uu=*vuu, &div=*vdiv, &ja=*vja;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxc = IDX(i, j, k, pdm_shap);
        REAL UE = uu(idxc                    , 0);
        REAL UW = uu(IDX(i-1, j, k, pdm_shap), 0);
        REAL VN = uu(idxc                    , 1);
        REAL VS = uu(IDX(i, j-1, k, pdm_shap), 1);
        REAL WT = uu(idxc                    , 2);
        REAL WB = uu(IDX(i, j, k-1, pdm_shap), 2);
        div(idxc) = (UE - UW + VN - VS + WT - WB) / ja(idxc);
    }
}

__global__ void kernel_Vortcity(
    const MatrixFrame<REAL> *vu,
    const MatrixFrame<REAL> *vkx,
    const MatrixFrame<REAL> *vvrt,
    INT3              pdm_shap,
    INT3              map_shap,
    INT3              map_offset
) {
    const MatrixFrame<REAL> &u = *vu, &kx = *vkx, &vrt = *vvrt;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap[0] && j < map_shap[1] && k < map_shap[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxc = IDX(i, j, k, pdm_shap);
        REAL kxx = kx(idxc, 0);
        REAL kxy = kx(idxc, 1);
        REAL kxz = kx(idxc, 2);
        REAL dvdx = 0.5 * kxx * (u(IDX(i+1,j,k,pdm_shap),1) - u(IDX(i-1,j,k,pdm_shap),1));
        REAL dwdx = 0.5 * kxx * (u(IDX(i+1,j,k,pdm_shap),2) - u(IDX(i-1,j,k,pdm_shap),2));
        REAL dudy = 0.5 * kxy * (u(IDX(i,j+1,k,pdm_shap),0) - u(IDX(i,j-1,k,pdm_shap),0));
        REAL dwdy = 0.5 * kxy * (u(IDX(i,j+1,k,pdm_shap),2) - u(IDX(i,j-1,k,pdm_shap),2));
        REAL dudz = 0.5 * kxz * (u(IDX(i,j,k+1,pdm_shap),0) - u(IDX(i,j,k-1,pdm_shap),0));
        REAL dvdz = 0.5 * kxz * (u(IDX(i,j,k+1,pdm_shap),1) - u(IDX(i,j,k-1,pdm_shap),1));
        vrt(idxc, 0) = dwdy - dvdz;
        vrt(idxc, 1) = dudz - dwdx;
        vrt(idxc, 2) = dvdx - dudy;
    }
}

void FalmCFDDevCall::FSPseudoU(
    Matrix<REAL> &un,
    Matrix<REAL> &u,
    Matrix<REAL> &uu,
    Matrix<REAL> &ua,
    Matrix<REAL> &nut,
    Matrix<REAL> &kx,
    Matrix<REAL> &g,
    Matrix<REAL> &ja,
    Matrix<REAL> &ff,
    REAL dt,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    STREAM        stream
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
    Matrix<REAL> &u,
    Matrix<REAL> &uc,
    Matrix<REAL> &kx,
    Matrix<REAL> &ja,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    STREAM        stream
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
    Matrix<REAL> &uu,
    Matrix<REAL> &uc,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    STREAM        stream
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
    Matrix<REAL> &u,
    Matrix<REAL> &ua,
    Matrix<REAL> &p,
    Matrix<REAL> &kx,
    REAL dt,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    STREAM        stream
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
    Matrix<REAL> &uu,
    Matrix<REAL> &uua,
    Matrix<REAL> &p,
    Matrix<REAL> &g,
    REAL dt,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    STREAM        stream
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
    Matrix<REAL> &u,
    Matrix<REAL> &nut,
    Matrix<REAL> &x,
    Matrix<REAL> &kx,
    Matrix<REAL> &ja,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    STREAM        stream
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
    Matrix<REAL> &uu,
    Matrix<REAL> &div,
    Matrix<REAL> &ja,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    STREAM        stream
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
    Matrix<REAL> &u,
    Matrix<REAL> &kx,
    Matrix<REAL> &vrt,
    Region       &pdm,
    const Region &map,
    dim3          block_dim,
    STREAM        stream
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
