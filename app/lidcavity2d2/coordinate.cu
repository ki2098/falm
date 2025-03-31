#include "coordinate.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

namespace LidCavity2d2 {

using namespace Falm;

__global__ void kernel_setCoord(
    Real               side_lenth,
    Int                side_n_cell,
    Int3              pdm_shape,
    Int3              pdm_offset,
    Int                gc,
    const MatrixFrame<Real> *vx,
    const MatrixFrame<Real> *vh,
    const MatrixFrame<Real> *vkx,
    const MatrixFrame<Real> *vg,
    const MatrixFrame<Real> *vja
) {
    const MatrixFrame<Real> &x=*vx, &h=*vh, &kx=*vkx, &g=*vg, &ja=*vja;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] && j < pdm_shape[1] && k < pdm_shape[2]) {
        Int idx = IDX(i, j, k, pdm_shape);
        Real pitch = side_lenth / side_n_cell;
        Real dkdx = 1.0 / pitch;
        Real vol = pitch * pitch * pitch;
        x(idx, 0) = (i + pdm_offset[0] - gc + 0.5) * pitch;
        x(idx, 1) = (j + pdm_offset[1] - gc + 0.5) * pitch;
        x(idx, 2) = (k + pdm_offset[2] - gc      ) * pitch;
        h(idx, 0) = h(idx, 1) = h(idx, 2) = pitch;
        kx(idx, 0) = kx(idx, 1) = kx(idx, 2) = dkdx;
        ja(idx) = vol;
        g(idx, 0) = g(idx, 1) = g(idx, 2) = vol * dkdx * dkdx;
    }
}

void setCoord(
    Real          side_lenth,
    Int           side_n_cell,
    Region       &pdm,
    Int           gc,
    Matrix<Real> &x,
    Matrix<Real> &h,
    Matrix<Real> &kx,
    Matrix<Real> &g,
    Matrix<Real> &ja,
    dim3          block_dim
) {
    x.alloc(pdm.shape, 3, HDC::Device);
    h.alloc(pdm.shape, 3, HDC::Device);
    kx.alloc(pdm.shape, 3, HDC::Device);
    g.alloc(pdm.shape, 3, HDC::Device);
    ja.alloc(pdm.shape, 1, HDC::Device);

    dim3 grid_dim(
        (pdm.shape[0] + block_dim.x - 1) / block_dim.x,
        (pdm.shape[1] + block_dim.y - 1) / block_dim.y,
        (pdm.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_setCoord<<<grid_dim, block_dim, 0, 0>>>(
        side_lenth,
        side_n_cell,
        pdm.shape,
        pdm.offset,
        gc,
        x.devptr,
        h.devptr,
        kx.devptr,
        g.devptr,
        ja.devptr
    );
}

}