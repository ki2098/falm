#include "coordinate.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

using namespace Falm;

namespace LidCavity2d {

__global__ void kernel_setCoord(
    REAL               side_lenth,
    INT                side_n_cell,
    INTx3              pdm_shape,
    MatrixFrame<REAL> &x,
    MatrixFrame<REAL> &h,
    MatrixFrame<REAL> &kx,
    MatrixFrame<REAL> &g,
    MatrixFrame<REAL> &ja
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x && j < pdm_shape.y && k < pdm_shape.z) {
        INT idx = IDX(i, j, k, pdm_shape);
        REAL pitch = side_lenth / side_n_cell;
        REAL dkdx = 1.0 / pitch;
        REAL vol = pitch * pitch * pitch;
        x(idx, 0) = (i + 1 - Gd) * pitch;
        x(idx, 1) = (j + 1 - Gd) * pitch;
        x(idx, 2) = (k     - Gd) * pitch;
        h(idx, 0) = h(idx, 1) = h(idx, 2) = pitch;
        kx(idx, 0) = kx(idx, 1) = kx(idx, 2) = dkdx;
        ja(idx) = vol;
        g(idx, 0) = g(idx, 1) = g(idx, 2) = vol * dkdx * dkdx;
    }
}

void setCoord(
    REAL          side_lenth,
    INT           side_n_cell,
    Mapper       &pdm,
    Matrix<REAL> &x,
    Matrix<REAL> &h,
    Matrix<REAL> &kx,
    Matrix<REAL> &g,
    Matrix<REAL> &ja,
    dim3          block_dim
) {
    pdm = Mapper(
        INTx3{side_n_cell - 1 + Gdx2, side_n_cell - 1 + Gdx2, 1 + Gdx2},
        INTx3{0, 0, 0}
    );
    x.alloc(pdm.shape, 3, HDCType::Device);
    h.alloc(pdm.shape, 3, HDCType::Device);
    kx.alloc(pdm.shape, 3, HDCType::Device);
    g.alloc(pdm.shape, 3, HDCType::Device);
    ja.alloc(pdm.shape, 1, HDCType::Device);

    dim3 grid_dim(
        (pdm.shape.x + block_dim.x - 1) / block_dim.x,
        (pdm.shape.y + block_dim.y - 1) / block_dim.y,
        (pdm.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_setCoord<<<grid_dim, block_dim, 0, 0>>>(
        side_lenth,
        side_n_cell,
        pdm.shape,
        *(x.devptr),
        *(h.devptr),
        *(kx.devptr),
        *(g.devptr),
        *(ja.devptr)
    );
}

}