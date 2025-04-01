#include "poissonDev.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

namespace LidCavity2d2 {

using namespace Falm;

__global__ void kernel_makePoissonMatrix(
    const MatrixFrame<Real> *va,
    const MatrixFrame<Real> *vg,
    const MatrixFrame<Real> *vja,
    Int3              global_shape,
    Int3              pdm_shape,
    Int3              pdm_offset,
    Int3              map_shape,
    Int3              map_offset,
    Int                gc
) {
    const MatrixFrame<Real> &a = *va, &g = *vg, &ja = *vja;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        Int gi, gj, gk;
        gi = i + pdm_offset[0];
        gj = j + pdm_offset[1];
        gk = k + pdm_offset[2];
        Real ac, ae, aw, an, as;
        ac = ae = aw = an = as = 0.0;
        Int idxcc = IDX(i  , j  , k  , pdm_shape);
        Int idxe1 = IDX(i+1, j  , k  , pdm_shape);
        Int idxw1 = IDX(i-1, j  , k  , pdm_shape);
        Int idxn1 = IDX(i  , j+1, k  , pdm_shape);
        Int idxs1 = IDX(i  , j-1, k  , pdm_shape);
        Int idxt1 = IDX(i  , j  , k+1, pdm_shape);
        Int idxb1 = IDX(i  , j  , k-1, pdm_shape);
        Real gxcc  =  g(idxcc, 0);
        Real gxe1  =  g(idxe1, 0);
        Real gxw1  =  g(idxw1, 0);
        Real gycc  =  g(idxcc, 1);
        Real gyn1  =  g(idxn1, 1);
        Real gys1  =  g(idxs1, 1);
        Real gzcc  =  g(idxcc, 2);
        Real gzt1  =  g(idxt1, 2);
        Real gzb1  =  g(idxb1, 2);
        Real jacob = ja(idxcc);
        Real coefficient;

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
    Matrix<Real> &a,
    Matrix<Real> &g,
    Matrix<Real> &ja,
    Region       &global,
    Region       &pdm,
    Int           gc,
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