#include "../FalmCFDL1.h"
#include "../FalmCFDScheme.h"
#include "devutil.cuh"

namespace Falm {

__global__ void kernel_Cartesian_CalcPseudoU(
    MatrixFrame<REAL> &u,
    MatrixFrame<REAL> &uu,
    MatrixFrame<REAL> &ua,
    MatrixFrame<REAL> &nut,
    MatrixFrame<REAL> &kx,
    MatrixFrame<REAL> &g,
    MatrixFrame<REAL> &jac,
    MatrixFrame<REAL> &ff,
    REAL               ReI,
    REAL               dt,
    INTx3              proc_shape,
    INTx3              map_shap,
    INTx3              map_offset
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap.x && j < map_shap.y && k < map_shap.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idxcc = IDX(i  , j  , k  , proc_shape);
        INT idxe1 = IDX(i+1, j  , k  , proc_shape);
        INT idxw1 = IDX(i-1, j  , k  , proc_shape);
        INT idxn1 = IDX(i  , j+1, k  , proc_shape);
        INT idxs1 = IDX(i  , j-1, k  , proc_shape);
        INT idxt1 = IDX(i  , j  , k+1, proc_shape);
        INT idxb1 = IDX(i  , j  , k-1, proc_shape);
        INT idxe2 = IDX(i+2, j  , k  , proc_shape);
        INT idxw2 = IDX(i-2, j  , k  , proc_shape);
        INT idxn2 = IDX(i  , j+2, k  , proc_shape);
        INT idxs2 = IDX(i  , j-2, k  , proc_shape);
        INT idxt2 = IDX(i  , j  , k+2, proc_shape);
        INT idxb2 = IDX(i  , j  , k-2, proc_shape);
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
        REAL ja    = jac(idxcc);

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
        adv = Riam3rdUpwind(
            ucc,
            ue1, ue2, uw1, uw2,
            un1, un2, us1, us2,
            ut1, ut2, ub1, ub2,
            Uabs, Vabs, Wabs,
            UE, UW, VN, VS, WT, WB,
            ja
        );
        vis = Diffusion(
            ReI,
            ucc, ue1, uw1, un1, us1, ut1, ub1,
            nutcc, nute1, nutw1, nutn1, nuts1, nutt1, nutb1,
            gxcc, gxe1, gxw1,
            gycc, gyn1, gys1,
            gzcc, gzt1, gzb1,
            ja
        );
        ua(idxcc, d) = ucc + dt * (- adv + vis + ff(idxcc, d));

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
        adv = Riam3rdUpwind(
            ucc,
            ue1, ue2, uw1, uw2,
            un1, un2, us1, us2,
            ut1, ut2, ub1, ub2,
            Uabs, Vabs, Wabs,
            UE, UW, VN, VS, WT, WB,
            ja
        );
        vis = Diffusion(
            ReI,
            ucc, ue1, uw1, un1, us1, ut1, ub1,
            nutcc, nute1, nutw1, nutn1, nuts1, nutt1, nutb1,
            gxcc, gxe1, gxw1,
            gycc, gyn1, gys1,
            gzcc, gzt1, gzb1,
            ja
        );
        ua(idxcc, d) = ucc + dt * (- adv + vis + ff(idxcc, d));

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
        adv = Riam3rdUpwind(
            ucc,
            ue1, ue2, uw1, uw2,
            un1, un2, us1, us2,
            ut1, ut2, ub1, ub2,
            Uabs, Vabs, Wabs,
            UE, UW, VN, VS, WT, WB,
            ja
        );
        vis = Diffusion(
            ReI,
            ucc, ue1, uw1, un1, us1, ut1, ub1,
            nutcc, nute1, nutw1, nutn1, nuts1, nutt1, nutb1,
            gxcc, gxe1, gxw1,
            gycc, gyn1, gys1,
            gzcc, gzt1, gzb1,
            ja
        );
        ua(idxcc, d) = ucc + dt * (- adv + vis + ff(idxcc, d));
    }
}

__global__ void kernel_Cartesian_UtoCU (
    MatrixFrame<REAL> &u,
    MatrixFrame<REAL> &uc,
    MatrixFrame<REAL> &kx,
    MatrixFrame<REAL> &jac,
    INTx3              proc_shape,
    INTx3              map_shap,
    INTx3              map_offset
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap.x && j < map_shap.y && k < map_shap.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, proc_shape);
        REAL ja = jac(idx);
        uc(idx, 0) = ja * kx(idx, 0) * u(idx, 0);
        uc(idx, 1) = ja * kx(idx, 1) * u(idx, 1);
        uc(idx, 2) = ja * kx(idx, 2) * u(idx, 2);
    }
}

void L1Explicit::L0Dev_Cartesian_FSCalcPseudoU(
    Matrix<REAL> &u,
    Matrix<REAL> &uu,
    Matrix<REAL> &ua,
    Matrix<REAL> &nut,
    Matrix<REAL> &kx,
    Matrix<REAL> &g,
    Matrix<REAL> &jac,
    Matrix<REAL> &ff,
    Mapper         &proc_domain,
    Mapper         &map,
    dim3            block_dim
) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_Cartesian_CalcPseudoU<<<grid_dim, block_dim, 0, 0>>>(
        *(u.devptr),
        *(uu.devptr),
        *(ua.devptr),
        *(nut.devptr),
        *(kx.devptr),
        *(g.devptr),
        *(jac.devptr),
        *(ff.devptr),
        ReI,
        dt,
        proc_domain.shape,
        map.shape,
        map.offset
    );
}

void L1Explicit::L0Dev_Cartesian_UtoCU(
    Matrix<REAL> &u,
    Matrix<REAL> &uc,
    Matrix<REAL> &kx,
    Matrix<REAL> &jac,
    Mapper         &proc_domain,
    Mapper         &map,
    dim3            block_dim
) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_Cartesian_UtoCU<<<grid_dim, block_dim, 0, 0>>>(
        *(u.devptr),
        *(uc.devptr),
        *(kx.devptr),
        *(jac.devptr),
        proc_domain.shape,
        map.shape,
        map.offset
    );
}

}