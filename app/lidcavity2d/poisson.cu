#include "poisson.h"
#include "../../src/MVL1.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

using namespace Falm;

namespace LidCavity2d {

__global__ void kernel_makePoissonMatrix(
    const MatrixFrame<REAL> *va,
    const MatrixFrame<REAL> *vg,
    const MatrixFrame<REAL> *vja,
    INTx3              pdm_shape,
    INTx3              map_shape,
    INTx3              map_offset,
    INT gc
) {
    const MatrixFrame<REAL> &a=*va, &g=*vg, &ja=*vja;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x & j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        REAL ac, ae, aw, an, as;
        ac = ae = aw = an = as = 0.0;
        INT idxcc = IDX(i  , j  , k  , pdm_shape);
        INT idxe1 = IDX(i+1, j  , k  , pdm_shape);
        INT idxw1 = IDX(i-1, j  , k  , pdm_shape);
        INT idxn1 = IDX(i  , j+1, k  , pdm_shape);
        INT idxs1 = IDX(i  , j-1, k  , pdm_shape);
        INT idxt1 = IDX(i  , j  , k+1, pdm_shape);
        INT idxb1 = IDX(i  , j  , k-1, pdm_shape);
        REAL gxcc  =  g(idxcc, 0);
        REAL gxe1  =  g(idxe1, 0);
        REAL gxw1  =  g(idxw1, 0);
        REAL gycc  =  g(idxcc, 1);
        REAL gyn1  =  g(idxn1, 1);
        REAL gys1  =  g(idxs1, 1);
        REAL gzcc  =  g(idxcc, 2);
        REAL gzt1  =  g(idxt1, 2);
        REAL gzb1  =  g(idxb1, 2);
        REAL jacob = ja(idxcc);
        REAL coefficient;

        coefficient = 0.5 * (gxcc + gxe1) / jacob;
        if (i < pdm_shape.x - gc - 1) {
            ac -= coefficient;
            ae = coefficient;
        }

        coefficient = 0.5 * (gxcc + gxw1) / jacob;
        if (i > gc) {
            ac -= coefficient;
            aw = coefficient;
        }

        coefficient = 0.5 * (gycc + gyn1) / jacob;
        if (j < pdm_shape.y - gc - 1) {
            ac -= coefficient;
            an = coefficient;
        }

        coefficient = 0.5 * (gycc + gys1) / jacob;
        if (j > gc) {
            ac -= coefficient;
            as = coefficient;
        }

        a(idxcc, 0) = ac;
        a(idxcc, 1) = ae;
        a(idxcc, 2) = aw;
        a(idxcc, 3) = an;
        a(idxcc, 4) = as;
    }
}

REAL makePoissonMatrix(
    Matrix<REAL> &a,
    Matrix<REAL> &g,
    Matrix<REAL> &ja,
    Region       &pdm,
    INT gc,
    dim3          block_dim
) {
    Region map(pdm.shape, gc);
    dim3 grid_dim(
        (pdm.shape.x + block_dim.x - 1) / block_dim.x,
        (pdm.shape.y + block_dim.y - 1) / block_dim.y,
        (pdm.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_makePoissonMatrix<<<grid_dim, block_dim, 0, 0>>>(
        a.devptr,
        g.devptr,
        ja.devptr,
        pdm.shape,
        map.shape,
        map.offset,
        gc
    );
    REAL maxdiag = L1Dev_MaxDiag(a, pdm, gc, block_dim);
    L1Dev_ScaleMatrix(a, 1.0 / maxdiag, block_dim);
    return maxdiag;
}

__global__ void kernel_makePoissonRHS(
    const MatrixFrame<REAL> *vp,
    const MatrixFrame<REAL> *vrhs,
    const MatrixFrame<REAL> *vg,
    const MatrixFrame<REAL> *vja,
    INTx3              pdm_shape,
    INTx3              map_shape,
    INTx3              map_offset
) {
    const MatrixFrame<REAL> &p=*vp, &rhs=*vrhs, &g=*vg, &ja=*vja;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x & j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idxcc = IDX(i  , j  , k  , pdm_shape);
        INT idxe1 = IDX(i+1, j  , k  , pdm_shape);
        INT idxw1 = IDX(i-1, j  , k  , pdm_shape);
        INT idxn1 = IDX(i  , j+1, k  , pdm_shape);
        INT idxs1 = IDX(i  , j-1, k  , pdm_shape);
        INT idxt1 = IDX(i  , j  , k+1, pdm_shape);
        INT idxb1 = IDX(i  , j  , k-1, pdm_shape);
        REAL gxcc  =  g(idxcc, 0);
        REAL gxe1  =  g(idxe1, 0);
        REAL gxw1  =  g(idxw1, 0);
        REAL gycc  =  g(idxcc, 1);
        REAL gyn1  =  g(idxn1, 1);
        REAL gys1  =  g(idxs1, 1);
        REAL gzcc  =  g(idxcc, 2);
        REAL gzt1  =  g(idxt1, 2);
        REAL gzb1  =  g(idxb1, 2);
        REAL  pcc  =  p(idxcc);
        REAL  pe1  =  p(idxe1);
        REAL  pw1  =  p(idxw1);
        REAL  pn1  =  p(idxn1);
        REAL  ps1  =  p(idxs1);
        REAL  pt1  =  p(idxt1);
        REAL  pb1  =  p(idxb1);
        REAL jacob = ja(idxcc);
        // if (i == pdm_shape.x - gc - 1) {
        //     rhs(idxcc) -= pe1 * 0.5 * (gxcc + gxe1) / jacob;
        // }
        // if (i == gc) {
        //     rhs(idxcc) -= pw1 * 0.5 * (gxcc + gxw1) / jacob;
        // }
        // if (j == pdm_shape.y - gc - 1) {
        //     rhs(idxcc) -= pn1 * 0.5 * (gycc + gyn1) / jacob;
        // }
        // if (j == gc) {
        //     rhs(idxcc) -= ps1 * 0.5 * (gycc + gys1) / jacob;
        // }
    } 
}

void makePoissonRHS(
    Matrix<REAL> &p,
    Matrix<REAL> &rhs,
    Matrix<REAL> &g,
    Matrix<REAL> &ja,
    REAL          maxdiag,
    Region       &pdm,
    INT gc,
    dim3          block_dim
) {
    Region map(pdm.shape, gc);
    dim3 grid_dim(
        (pdm.shape.x + block_dim.x - 1) / block_dim.x,
        (pdm.shape.y + block_dim.y - 1) / block_dim.y,
        (pdm.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_makePoissonRHS<<<grid_dim, block_dim, 0, 0>>>(
        p.devptr,
        rhs.devptr,
        g.devptr,
        ja.devptr,
        pdm.shape,
        map.shape,
        map.offset
    );
    L1Dev_ScaleMatrix(rhs, 1.0 / maxdiag, block_dim);
}

}