#include "poissonDev.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

namespace LidCavity2d2 {

using namespace Falm;

__global__ void kernel_makePoissonMatrix(
    const MatrixFrame<REAL> *va,
    const MatrixFrame<REAL> *vg,
    const MatrixFrame<REAL> *vja,
    INT3              global_shape,
    INT3              pdm_shape,
    INT3              pdm_offset,
    INT3              map_shape,
    INT3              map_offset,
    INT                gc
) {
    const MatrixFrame<REAL> &a = *va, &g = *vg, &ja = *vja;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT gi, gj, gk;
        gi = i + pdm_offset[0];
        gj = j + pdm_offset[1];
        gk = k + pdm_offset[2];
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
        if (gi < global_shape[0] - gc - 1) {
            ac -= coefficient;
            ae  = coefficient;
        }

        coefficient = 0.5 * (gxcc + gxw1) / jacob;
        if (gi > gc) {
            ac -= coefficient;
            aw  = coefficient;
        }

        coefficient = 0.5 * (gycc + gyn1) / jacob;
        if (gj < global_shape[1] - gc - 1) {
            ac -= coefficient;
            an  = coefficient;
        }

        coefficient = 0.5 * (gycc + gys1) / jacob;
        if (gj > gc) {
            ac -= coefficient;
            as  = coefficient;
        }

        a(idxcc, 0) = ac;
        a(idxcc, 1) = aw;
        a(idxcc, 2) = ae;
        a(idxcc, 3) = as;
        a(idxcc, 4) = an;
    }
}

void dev_makePoissonMatrix(
    Matrix<REAL> &a,
    Matrix<REAL> &g,
    Matrix<REAL> &ja,
    Region       &global,
    Region       &pdm,
    INT           gc,
    dim3          block_dim
) {
    Region map(pdm.shape, gc);
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_makePoissonMatrix<<<grid_dim, block_dim, 0, 0>>>(
        a.devptr,
        g.devptr,
        ja.devptr,
        global.shape,
        pdm.shape,
        pdm.offset,
        map.shape,
        map.offset,
        gc
    );
}

}