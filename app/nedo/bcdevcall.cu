#include "bcdevcall.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

using namespace Falm;

/** inlet */
__global__ void kernel_ubc_xminus(const MatrixFrame<REAL> *vu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - gc*2 && k < shape[2] - gc*2) {
        i += gc - 1; j += gc; k += gc;
        INT idxo1 = IDX(i  ,j,k,shape);
        INT idxo2 = IDX(i-1,j,k,shape);
        const REAL3 ubc = {1.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(idxo1, d) = ubc[d];
            u(idxo2, d) = ubc[d];
        }
    }
}

/** outlet */
__global__ void kernel_ubc_xplus(const MatrixFrame<REAL> *vu, const MatrixFrame<REAL> *vuprev, const MatrixFrame<REAL> *vx, REAL dt, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &u = *vu, &uprev = *vuprev, &x = *vx;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - gc*2 && k < shape[2] - gc*2) {
        i += shape[0] - gc; j += gc; k += gc;
        INT idxo1 = IDX(i  ,j,k,shape);
        INT idxo2 = IDX(i+1,j,k,shape);
        INT idxi1 = IDX(i-1,j,k,shape);
        INT idxi2 = IDX(i-2,j,k,shape);
        REAL dxi = 1.0 / (x(idxo1, 0) - x(idxi1, 0));
        for (INT d = 0; d < 3; d ++) {
            REAL du = 0.5 * dxi * (3*uprev(idxo1, d) - 4*uprev(idxi1, d) + uprev(idxi2, d));
            REAL ubc = uprev(idxo1, d) - dt * du;
            u(idxo1, d) =     ubc;
            u(idxo2, d) = 2 * ubc - u(idxi1, d);
        }
    }
}

/** slip */
__global__ void kernel_ubc_yminus(const MatrixFrame<REAL> *vu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < 1 && k < shape[2] - gc*2) {
        i += gc; j += gc - 1; k += gc;
        INT idxo1 = IDX(i,j  ,k,shape);
        INT idxo2 = IDX(i,j-1,k,shape);
        INT idxi1 = IDX(i,j+1,k,shape);
        REAL3 ubc = {u(idxi1, 0), 0.0, u(idxi1, 2)};
        for (INT d = 0; d < 3; d ++) {
            u(idxo1, d) =     ubc[d];
            u(idxo2, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

/** slip */
__global__ void kernel_ubc_yplus(const MatrixFrame<REAL> *vu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < 1 && k < shape[2] - gc*2) {
        i += gc; j += shape[1] - gc; k += gc;
        INT idxo1 = IDX(i,j  ,k,shape);
        INT idxo2 = IDX(i,j+1,k,shape);
        INT idxi1 = IDX(i,j-1,k,shape);
        REAL3 ubc = {u(idxi1, 0), 0.0, u(idxi1, 2)};
        for (INT d = 0; d < 3; d ++) {
            u(idxo1, d) =     ubc[d];
            u(idxo2, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

/** wall */
__global__ void kernel_ubc_zminus(const MatrixFrame<REAL> *vu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < shape[1] - gc * 2 && k < 1) {
        i += gc; j += gc; k += gc-1;
        INT idxo1 = IDX(i,j,k  ,shape);
        INT idxo2 = IDX(i,j,k-1,shape);
        INT idxi1 = IDX(i,j,k+1,shape);
        REAL3 ubc = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(idxo1, d) =     ubc[d];
            u(idxo2, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

/** slip */
__global__ void kernel_ubc_zplus(const MatrixFrame<REAL> *vu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < shape[1] - gc * 2 && k < 1) {
        i += gc; j += gc; k += shape[2] - gc;
        INT idxo1 = IDX(i,j,k  ,shape);
        INT idxo2 = IDX(i,j,k+1,shape);
        INT idxi1 = IDX(i,j,k-1,shape);
        REAL3 ubc = {u(idxi1, 0), u(idxi1, 1), 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(idxo1, d) =     ubc[d];
            u(idxo2, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

void ubc_xminus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM s) {
    dim3 grid(1, (pdm.shape[1] - gc*2 + 7) / 8, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_ubc_xminus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void ubc_xplus(Matrix<REAL> &u, Matrix<REAL> &uprev, Matrix<REAL> &x, REAL dt, Region &pdm, INT gc, STREAM s) {
    dim3 grid(1, (pdm.shape[1] - gc*2 + 7) / 8, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_ubc_xplus<<<grid, block, 0, s>>>(u.devptr, uprev.devptr, x.devptr, dt, pdm.shape, gc);
}

void ubc_yminus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, 1, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_ubc_yminus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void ubc_yplus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, 1, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_ubc_yplus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void ubc_zminus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, (pdm.shape[1] - gc*2 + 7) / 8, 1);
    dim3 block(8, 8, 1);
    kernel_ubc_zminus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void ubc_zplus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, (pdm.shape[1] - gc*2 + 7) / 8, 1);
    dim3 block(8, 8, 1);
    kernel_ubc_zplus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

__global__ void kernel_pbc_xminus(const MatrixFrame<REAL> *vp, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - gc*2 && k < shape[2] - gc*2) {
        i += gc - 1; j += gc; k += gc;
        p(IDX(i,j,k,shape)) = p(IDX(i+1,j,k,shape));
    }
}

__global__ void kernel_pbc_xplus(const MatrixFrame<REAL> *vp, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - gc*2 && k < shape[2] - gc*2) {
        i += shape[0] - gc; j += gc; k += gc;
        p(IDX(i,j,k,shape)) = 0.0;
    }
}

__global__ void kernel_pbc_yminus(const MatrixFrame<REAL> *vp, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < 1 && k < shape[2] - gc*2) {
        i += gc; j += gc - 1; k += gc;
        p(IDX(i,j,k,shape)) = p(IDX(i,j+1,k,shape));
    }
}

__global__ void kernel_pbc_yplus(const MatrixFrame<REAL> *vp, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < 1 && k < shape[2] - gc*2) {
        i += gc; j += shape[1] - gc; k += gc;
        p(IDX(i,j,k,shape)) = p(IDX(i,j-1,k,shape));
    }
}

__global__ void kernel_pbc_zminus(const MatrixFrame<REAL> *vp, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < shape[1] - gc * 2 && k < 1) {
        i += gc; j += gc; k += gc-1;
        p(IDX(i,j,k,shape)) = p(IDX(i,j,k+1,shape));
    }
}

__global__ void kernel_pbc_zplus(const MatrixFrame<REAL> *vp, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < shape[1] - gc * 2 && k < 1) {
        i += gc; j += gc; k += shape[2] - gc;
        p(IDX(i,j,k,shape)) = p(IDX(i,j,k-1,shape));
    }
}

void pbc_xminus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM s) {
    dim3 grid(1, (pdm.shape[1] - gc*2 + 7) / 8, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_pbc_xminus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_xplus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM s) {
    dim3 grid(1, (pdm.shape[1] - gc*2 + 7) / 8, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_pbc_xplus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_yminus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, 1, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_pbc_yminus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_yplus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, 1, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_pbc_yplus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_zminus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, (pdm.shape[1] - gc*2 + 7) / 8, 1);
    dim3 block(8, 8, 1);
    kernel_pbc_zminus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_zplus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, (pdm.shape[1] - gc*2 + 7) / 8, 1);
    dim3 block(8, 8, 1);
    kernel_pbc_zplus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}