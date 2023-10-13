#include "poissonDev.h"
#include "../../src/MVL1.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

namespace LidCavity2d2 {

using namespace Falm;

__global__ void kernel_makePoissonMatrix(
    MatrixFrame<REAL> &a,
    MatrixFrame<REAL> &g,
    MatrixFrame<REAL> &ja,
    INTx3              global_shape,
    INTx3              pdm_shape,
    INTx3              pdm_offset,
    INTx3              map_shape,
    INTx3              map_offset
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT gi, gj, gk;
        gi = i + pdm_offset.x;
        gj = j + pdm_offset.y;
        gk = k + pdm_offset.z;
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
        if (gi < global_shape.x - Gd - 1) {
            ac -= coefficient;
            ae  = coefficient;
        }

        coefficient = 0.5 * (gxcc + gxw1) / jacob;
        if (gi > Gd) {
            ac -= coefficient;
            aw  = coefficient;
        }

        coefficient = 0.5 * (gycc + gyn1) / jacob;
        if (gj < global_shape.y - Gd - 1) {
            ac -= coefficient;
            an  = coefficient;
        }

        coefficient = 0.5 * (gycc + gys1) / jacob;
        if (gj > Gd) {
            ac -= coefficient;
            as  = coefficient;
        }

        a(idxcc, 0) = ac;
        a(idxcc, 1) = ae;
        a(idxcc, 2) = aw;
        a(idxcc, 3) = an;
        a(idxcc, 4) = as;
    }
}

void dev_makePoissonMatrix(
    Matrix<REAL> &a,
    Matrix<REAL> &g,
    Matrix<REAL> &ja,
    Mapper       &global,
    Mapper       &pdm,
    dim3          block_dim
) {
    Mapper map(pdm, Gd);
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_makePoissonMatrix<<<grid_dim, block_dim, 0, 0>>>(
        *(a.devptr),
        *(g.devptr),
        *(ja.devptr),
        global.shape,
        pdm.shape,
        pdm.offset,
        map.shape,
        map.offset
    );
}

}