#include "boundaryDev.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

namespace TURBINE1 {

using namespace Falm;

__global__ void kernel_vbc_xminus(
    const MatrixFrame<REAL> *vu,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        REAL ubc[] = {1.0, 0.0, 0.0};
        INT idxcc = IDX(i  , j, k, pdm_shape);
        INT idxo1 = IDX(i-1, j, k, pdm_shape);
        INT idxi1 = IDX(i+1, j, k, pdm_shape);
        for (INT d = 0; d < 3; d ++) {
            u(idxcc, d) =     ubc[d];
            u(idxo1, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

__global__ void kernel_vbc_xplus(
    const MatrixFrame<REAL> *vu, const MatrixFrame<REAL> *vun, const MatrixFrame<REAL> *vx,
    REAL dt, INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &u = *vu, &un = *vun, &x = *vx;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += pdm_shape[0] - gc;
        j += gc;
        k += gc;

        INT idxi2 = IDX(i-2, j, k, pdm_shape);
        INT idxi1 = IDX(i-1, j, k, pdm_shape);
        INT idxcc = IDX(i  , j, k, pdm_shape);
        INT idxo1 = IDX(i+1, j, k, pdm_shape);
        REAL dxi  = 1.0 / (x(idxcc) - x(idxi1));
        for (INT d = 0; d < 3; d ++) {
            REAL du = 0.5 * dxi * (3 * un(idxcc, d) - 4 * un(idxi1, d) + un(idxi2, d));
            REAL ubc = un(idxcc, d) - dt * du;
            u(idxcc, d) =     ubc;
            u(idxo1, d) = 2 * ubc - u(idxi1, d);
        }
    }
}

__global__ void kernel_vbc_yminus(
    const MatrixFrame<REAL> *vu,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        INT idxcc = IDX(i, j  , k, pdm_shape);
        INT idxo1 = IDX(i, j-1, k, pdm_shape);
        INT idxi1 = IDX(i, j+1, k, pdm_shape);
        REAL ubc[] = {u(idxi1, 0), 0.0, u(idxi1, 2)};
        for (INT d = 0; d < 3; d ++) {
            u(idxcc, d) =     ubc[d];
            u(idxo1, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

__global__ void kernel_vbc_yplus(
    const MatrixFrame<REAL> *vu,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += pdm_shape[1] - gc;
        k += gc;
        INT idxcc = IDX(i, j  , k, pdm_shape);
        INT idxo1 = IDX(i, j+1, k, pdm_shape);
        INT idxi1 = IDX(i, j-1, k, pdm_shape);
        REAL ubc[] = {u(idxi1, 0), 0.0, u(idxi1, 2)};
        for (INT d = 0; d < 3; d ++) {
            u(idxcc, d) =     ubc[d];
            u(idxo1, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

__global__ void kernel_vbc_zminus(
    const MatrixFrame<REAL> *vu,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < pdm_shape[1] - (gc*2) && k < 1) {
        i += gc;
        j += gc;
        k += gc - 1;
        INT idxcc = IDX(i, j, k  , pdm_shape);
        INT idxo1 = IDX(i, j, k-1, pdm_shape);
        INT idxi1 = IDX(i, j, k+1, pdm_shape);
        REAL ubc[] = {u(idxi1, 0), u(idxi1, 1), 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(idxcc, d) =     ubc[d];
            u(idxo1, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

__global__ void kernel_vbc_zplus(
    const MatrixFrame<REAL> *vu,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < pdm_shape[1] - (gc*2) && k < 1) {
        i += gc;
        j += gc;
        k += pdm_shape[2] - gc;
        INT idxcc = IDX(i, j, k  , pdm_shape);
        INT idxo1 = IDX(i, j, k+1, pdm_shape);
        INT idxi1 = IDX(i, j, k-1, pdm_shape);
        REAL ubc[] = {u(idxi1, 0), u(idxi1, 1), 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(idxcc, d) =     ubc[d];
            u(idxo1, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

__global__ void kernel_pbc_xminus(
    const MatrixFrame<REAL> *vp,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i+1, j, k, pdm_shape));
    }
}

__global__ void kernel_pbc_xplus(
    const MatrixFrame<REAL> *vp,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += pdm_shape[0] - gc;
        j += gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = 0;
    }
}

__global__ void kernel_pbc_yminus(
    const MatrixFrame<REAL> *vp,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j+1, k, pdm_shape));
    }
}

__global__ void kernel_pbc_yplus(
    const MatrixFrame<REAL> *vp,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += pdm_shape[1] - gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j-1, k, pdm_shape));
    }
}

__global__ void kernel_pbc_zminus(
    const MatrixFrame<REAL> *vp,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < pdm_shape[1] - (gc*2) && k < 1) {
        i += gc;
        j += gc;
        k += gc - 1;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j, k+1, pdm_shape));
    }
}

__global__ void kernel_pbc_zplus(
    const MatrixFrame<REAL> *vp,
    INT3 pdm_shape, INT gc
) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < pdm_shape[1] - (gc*2) && k < 1) {
        i += gc;
        j += gc;
        k += pdm_shape[2] - gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j, k-1, pdm_shape));
    }
}

void vbc_xminus(
    Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim(1, 8, 8);
    kernel_vbc_xminus<<<grid_dim, block_dim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void vbc_xplus(
    Matrix<REAL> &u, Matrix<REAL> &un, Matrix<REAL> &x, REAL dt, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim(1, 8, 8);
    kernel_vbc_xplus<<<grid_dim, block_dim, 0, stream>>>(u.devptr, un.devptr, x.devptr, dt, pdm.shape, gc);
}

void vbc_yminus(
    Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim(8, 1, 8);
    kernel_vbc_yminus<<<grid_dim, block_dim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void vbc_yplus(
    Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim(8, 1, 8);
    kernel_vbc_yplus<<<grid_dim, block_dim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void vbc_zminus(
    Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim((pdm.shape[0] - (gc*2) + 7) / 8, (pdm.shape[1] - (gc*2) + 7) / 8, 1);
    dim3 block_dim(8, 8, 1);
    kernel_vbc_zminus<<<grid_dim, block_dim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void vbc_zplus(
    Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim((pdm.shape[0] - (gc*2) + 7) / 8, (pdm.shape[1] - (gc*2) + 7) / 8, 1);
    dim3 block_dim(8, 8, 1);
    kernel_vbc_zplus<<<grid_dim, block_dim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void pbc_xminus(
    Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim(1, 8, 8);
    kernel_pbc_xminus<<<grid_dim, block_dim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void pbc_xplus(
    Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim(1, 8, 8);
    kernel_pbc_xplus<<<grid_dim, block_dim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void pbc_yminus(
    Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim(8, 1, 8);
    kernel_pbc_yminus<<<grid_dim, block_dim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void pbc_yplus(
    Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim(8, 1, 8);
    kernel_pbc_yplus<<<grid_dim, block_dim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void pbc_zminus(
    Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim((pdm.shape[0] - (gc*2) + 7) / 8, (pdm.shape[1] - (gc*2) + 7) / 8, 1);
    dim3 block_dim(8, 8, 1);
    kernel_pbc_zminus<<<grid_dim, block_dim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void pbc_zplus(
    Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream
) {
    dim3 grid_dim((pdm.shape[0] - (gc*2) + 7) / 8, (pdm.shape[1] - (gc*2) + 7) / 8, 1);
    dim3 block_dim(8, 8, 1);
    kernel_pbc_zplus<<<grid_dim, block_dim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

}