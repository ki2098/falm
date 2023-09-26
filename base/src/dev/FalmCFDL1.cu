#include "../FalmCFDL1.h"
#include "../FalmCFDScheme.h"
#include "devutil.cuh"

namespace Falm {

__global__ void kernel_Cartesian_CalcPseudoU(
    MatrixFrame<double> &u,
    MatrixFrame<double> &uu,
    MatrixFrame<double> &ua,
    MatrixFrame<double> &nut,
    MatrixFrame<double> &kx,
    MatrixFrame<double> &g,
    MatrixFrame<double> &jac,
    MatrixFrame<double> &ff,
    double               ReI,
    double               dt,
    uint3                proc_shape,
    uint3                map_shap,
    uint3                map_offset
) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap.x && j < map_shap.y && k < map_shap.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idxcc = IDX(i  , j  , k  , proc_shape);
        unsigned int idxe1 = IDX(i+1, j  , k  , proc_shape);
        unsigned int idxw1 = IDX(i-1, j  , k  , proc_shape);
        unsigned int idxn1 = IDX(i  , j+1, k  , proc_shape);
        unsigned int idxs1 = IDX(i  , j-1, k  , proc_shape);
        unsigned int idxt1 = IDX(i  , j  , k+1, proc_shape);
        unsigned int idxb1 = IDX(i  , j  , k-1, proc_shape);
        unsigned int idxe2 = IDX(i+2, j  , k  , proc_shape);
        unsigned int idxw2 = IDX(i-2, j  , k  , proc_shape);
        unsigned int idxn2 = IDX(i  , j+2, k  , proc_shape);
        unsigned int idxs2 = IDX(i  , j-2, k  , proc_shape);
        unsigned int idxt2 = IDX(i  , j  , k+2, proc_shape);
        unsigned int idxb2 = IDX(i  , j  , k-2, proc_shape);
        unsigned int idxE  = idxcc;
        unsigned int idxW  = idxw1;
        unsigned int idxN  = idxcc;
        unsigned int idxS  = idxs1;
        unsigned int idxT  = idxcc;
        unsigned int idxB  = idxb1;
        double uc    = u(idxcc, 0);
        double vc    = u(idxcc, 1);
        double wc    = u(idxcc, 2);
        double Uabs  = fabs(uc * kx(idxcc, 0));
        double Vabs  = fabs(vc * kx(idxcc, 1));
        double Wabs  = fabs(wc * kx(idxcc, 2));
        double UE    = uu(idxE, 0);
        double UW    = uu(idxW, 0);
        double VN    = uu(idxN, 1);
        double VS    = uu(idxS, 1);
        double WT    = uu(idxT, 2);
        double WB    = uu(idxB, 2);
        double nutcc = nut(idxcc);
        double nute1 = nut(idxe1);
        double nutw1 = nut(idxw1);
        double nutn1 = nut(idxn1);
        double nuts1 = nut(idxs1);
        double nutt1 = nut(idxt1);
        double nutb1 = nut(idxb1);
        double gxcc  = g(idxcc, 0);
        double gxe1  = g(idxe1, 0);
        double gxw1  = g(idxw1, 0);
        double gycc  = g(idxcc, 1);
        double gyn1  = g(idxn1, 1);
        double gys1  = g(idxs1, 1);
        double gzcc  = g(idxcc, 2);
        double gzt1  = g(idxt1, 2);
        double gzb1  = g(idxb1, 2);
        double ja    = jac(idxcc);

        unsigned int d;
        double ucc;
        double ue1, ue2, uw1, uw2;
        double un1, un2, us1, us2;
        double ut1, ut2, ub1, ub2;
        double adv, vis;

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
    MatrixFrame<double> &u,
    MatrixFrame<double> &uc,
    MatrixFrame<double> &kx,
    MatrixFrame<double> &jac,
    uint3                proc_shape,
    uint3                map_shap,
    uint3                map_offset
) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shap.x && j < map_shap.y && k < map_shap.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idx = IDX(i, j, k, proc_shape);
        double ja = jac(idx);
        uc(idx, 0) = ja * kx(idx, 0) * u(idx, 0);
        uc(idx, 1) = ja * kx(idx, 1) * u(idx, 1);
        uc(idx, 2) = ja * kx(idx, 2) * u(idx, 2);
    }
}

void L1Explicit::L0Dev_Cartesian_FSCalcPseudoU(
    Matrix<double> &u,
    Matrix<double> &uu,
    Matrix<double> &ua,
    Matrix<double> &nut,
    Matrix<double> &kx,
    Matrix<double> &g,
    Matrix<double> &jac,
    Matrix<double> &ff,
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
    Matrix<double> &u,
    Matrix<double> &uc,
    Matrix<double> &kx,
    Matrix<double> &jac,
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