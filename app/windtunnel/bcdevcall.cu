#include "bcdevcall.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

using namespace Falm;

/** inlet */
__global__ void kernel_ubc_xminus(const MatrixFrame<Real> *vu, Real u_inlet, Int3 shape, Int gc) {
    const MatrixFrame<Real> &u = *vu;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - gc*2 && k < shape[2] - gc*2) {
        i += gc - 1; j += gc; k += gc;
        Int idxo1 = IDX(i  ,j,k,shape);
        Int idxo2 = IDX(i-1,j,k,shape);
        const Real3 ubc{{u_inlet, 0.0, 0.0}};
        for (Int d = 0; d < 3; d ++) {
            u(idxo1, d) = ubc[d];
            u(idxo2, d) = ubc[d];
        }
    }
}

/** outlet */
__global__ void kernel_ubc_xplus(const MatrixFrame<Real> *vu, const MatrixFrame<Real> *vuprev, const MatrixFrame<Real> *vx, Real dt, Int3 shape, Int gc) {
    const MatrixFrame<Real> &u = *vu, &uprev = *vuprev, &x = *vx;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - gc*2 && k < shape[2] - gc*2) {
        i += shape[0] - gc; j += gc; k += gc;
        Int idxo1 = IDX(i  ,j,k,shape);
        Int idxo2 = IDX(i+1,j,k,shape);
        Int idxi1 = IDX(i-1,j,k,shape);
        Int idxi2 = IDX(i-2,j,k,shape);
        Real dxi = 1.0 / (x(idxo1, 0) - x(idxi1, 0));
        for (Int d = 0; d < 3; d ++) {
            Real du = 0.5 * dxi * (3*uprev(idxo1, d) - 4*uprev(idxi1, d) + uprev(idxi2, d));
            Real ubc = uprev(idxo1, d) - dt * du;
            u(idxo1, d) =     ubc;
            u(idxo2, d) = 2 * ubc - u(idxi1, d);
        }
    }
}

/** slip */
__global__ void kernel_ubc_yminus(const MatrixFrame<Real> *vu, Int3 shape, Int gc) {
    const MatrixFrame<Real> &u = *vu;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < 1 && k < shape[2] - gc*2) {
        i += gc; j += gc - 1; k += gc;
        Int idxo1 = IDX(i,j  ,k,shape);
        Int idxo2 = IDX(i,j-1,k,shape);
        Int idxi1 = IDX(i,j+1,k,shape);
        Real3 ubc = {u(idxi1, 0), 0.0, u(idxi1, 2)};
        for (Int d = 0; d < 3; d ++) {
            u(idxo1, d) =     ubc[d];
            u(idxo2, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

/** slip */
__global__ void kernel_ubc_yplus(const MatrixFrame<Real> *vu, Int3 shape, Int gc) {
    const MatrixFrame<Real> &u = *vu;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < 1 && k < shape[2] - gc*2) {
        i += gc; j += shape[1] - gc; k += gc;
        Int idxo1 = IDX(i,j  ,k,shape);
        Int idxo2 = IDX(i,j+1,k,shape);
        Int idxi1 = IDX(i,j-1,k,shape);
        Real3 ubc = {u(idxi1, 0), 0.0, u(idxi1, 2)};
        for (Int d = 0; d < 3; d ++) {
            u(idxo1, d) =     ubc[d];
            u(idxo2, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

/** wall */
__global__ void kernel_ubc_zminus(const MatrixFrame<Real> *vu, Int3 shape, Int gc) {
    const MatrixFrame<Real> &u = *vu;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < shape[1] - gc * 2 && k < 1) {
        i += gc; j += gc; k += gc-1;
        Int idxo1 = IDX(i,j,k  ,shape);
        Int idxo2 = IDX(i,j,k-1,shape);
        Int idxi1 = IDX(i,j,k+1,shape);
        Real3 ubc = {0.0, 0.0, 0.0};
        for (Int d = 0; d < 3; d ++) {
            u(idxo1, d) =     ubc[d];
            u(idxo2, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

/** slip */
__global__ void kernel_ubc_zplus(const MatrixFrame<Real> *vu, Int3 shape, Int gc) {
    const MatrixFrame<Real> &u = *vu;
    Int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < shape[1] - gc * 2 && k < 1) {
        i += gc; j += gc; k += shape[2] - gc;
        Int idxo1 = IDX(i,j,k  ,shape);
        Int idxo2 = IDX(i,j,k+1,shape);
        Int idxi1 = IDX(i,j,k-1,shape);
        Real3 ubc = {u(idxi1, 0), u(idxi1, 1), 0.0};
        for (Int d = 0; d < 3; d ++) {
            u(idxo1, d) =     ubc[d];
            u(idxo2, d) = 2 * ubc[d] - u(idxi1, d);
        }
    }
}

void ubc_xminus(Matrix<Real> &u, Real u_inlet, Region &pdm, Int gc, Stream s) {
    dim3 grid(1, (pdm.shape[1] - gc*2 + 7) / 8, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_ubc_xminus<<<grid, block, 0, s>>>(u.devptr, u_inlet, pdm.shape, gc);
}

void ubc_xplus(Matrix<Real> &u, Matrix<Real> &uprev, Matrix<Real> &x, Real dt, Region &pdm, Int gc, Stream s) {
    dim3 grid(1, (pdm.shape[1] - gc*2 + 7) / 8, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_ubc_xplus<<<grid, block, 0, s>>>(u.devptr, uprev.devptr, x.devptr, dt, pdm.shape, gc);
}

void ubc_yminus(Matrix<Real> &u, Region &pdm, Int gc, Stream s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, 1, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_ubc_yminus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void ubc_yplus(Matrix<Real> &u, Region &pdm, Int gc, Stream s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, 1, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_ubc_yplus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void ubc_zminus(Matrix<Real> &u, Region &pdm, Int gc, Stream s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, (pdm.shape[1] - gc*2 + 7) / 8, 1);
    dim3 block(8, 8, 1);
    kernel_ubc_zminus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void ubc_zplus(Matrix<Real> &u, Region &pdm, Int gc, Stream s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, (pdm.shape[1] - gc*2 + 7) / 8, 1);
    dim3 block(8, 8, 1);
    kernel_ubc_zplus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

__global__ void kernel_pbc_xminus(const MatrixFrame<Real> *vp, Int3 shape, Int gc) {
    const MatrixFrame<Real> &p = *vp;
    Int i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - gc*2 && k < shape[2] - gc*2) {
        i += gc - 1; j += gc; k += gc;
        p(IDX(i,j,k,shape)) = p(IDX(i+1,j,k,shape));
    }
}

__global__ void kernel_pbc_xplus(const MatrixFrame<Real> *vp, Int3 shape, Int gc) {
    const MatrixFrame<Real> &p = *vp;
    Int i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - gc*2 && k < shape[2] - gc*2) {
        i += shape[0] - gc; j += gc; k += gc;
        p(IDX(i,j,k,shape)) = 0.0;
    }
}

__global__ void kernel_pbc_yminus(const MatrixFrame<Real> *vp, Int3 shape, Int gc) {
    const MatrixFrame<Real> &p = *vp;
    Int i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < 1 && k < shape[2] - gc*2) {
        i += gc; j += gc - 1; k += gc;
        p(IDX(i,j,k,shape)) = p(IDX(i,j+1,k,shape));
    }
}

__global__ void kernel_pbc_yplus(const MatrixFrame<Real> *vp, Int3 shape, Int gc) {
    const MatrixFrame<Real> &p = *vp;
    Int i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < 1 && k < shape[2] - gc*2) {
        i += gc; j += shape[1] - gc; k += gc;
        p(IDX(i,j,k,shape)) = p(IDX(i,j-1,k,shape));
    }
}

__global__ void kernel_pbc_zminus(const MatrixFrame<Real> *vp, Int3 shape, Int gc) {
    const MatrixFrame<Real> &p = *vp;
    Int i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < shape[1] - gc * 2 && k < 1) {
        i += gc; j += gc; k += gc-1;
        p(IDX(i,j,k,shape)) = p(IDX(i,j,k+1,shape));
    }
}

__global__ void kernel_pbc_zplus(const MatrixFrame<Real> *vp, Int3 shape, Int gc) {
    const MatrixFrame<Real> &p = *vp;
    Int i, j, k; 
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - gc*2 && j < shape[1] - gc * 2 && k < 1) {
        i += gc; j += gc; k += shape[2] - gc;
        p(IDX(i,j,k,shape)) = p(IDX(i,j,k-1,shape));
    }
}

void pbc_xminus(Matrix<Real> &p, Region &pdm, Int gc, Stream s) {
    dim3 grid(1, (pdm.shape[1] - gc*2 + 7) / 8, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_pbc_xminus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_xplus(Matrix<Real> &p, Region &pdm, Int gc, Stream s) {
    dim3 grid(1, (pdm.shape[1] - gc*2 + 7) / 8, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_pbc_xplus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_yminus(Matrix<Real> &p, Region &pdm, Int gc, Stream s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, 1, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_pbc_yminus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_yplus(Matrix<Real> &p, Region &pdm, Int gc, Stream s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, 1, (pdm.shape[2] - gc*2 + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_pbc_yplus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_zminus(Matrix<Real> &p, Region &pdm, Int gc, Stream s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, (pdm.shape[1] - gc*2 + 7) / 8, 1);
    dim3 block(8, 8, 1);
    kernel_pbc_zminus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_zplus(Matrix<Real> &p, Region &pdm, Int gc, Stream s) {
    dim3 grid((pdm.shape[0] - gc*2 + 7) / 8, (pdm.shape[1] - gc*2 + 7) / 8, 1);
    dim3 block(8, 8, 1);
    kernel_pbc_zplus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}