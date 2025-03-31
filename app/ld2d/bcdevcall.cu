#include "bcdevcall.h"
#include "../../src/dev/devutil.cuh"

using namespace Falm;
using namespace std;

__global__ void kernel_pbc_xplus(const MatrixFrame<REAL> *vp, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - (gc*2) && k < shape[2] - (gc*2)) {
        i += shape[0] - gc;
        j += gc;
        k += gc;
        p(IDX(i, j, k, shape)) = p(IDX(i-1, j, k, shape));
    }
}

__global__ void kernel_pbc_xminus(const MatrixFrame<REAL> *vp, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - (gc*2) && k < shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        p(IDX(i, j, k, shape)) = p(IDX(i+1, j, k, shape));
    }
}

__global__ void kernel_pbc_yplus(const MatrixFrame<REAL> *vp, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - (gc*2) && j < 1 && k < shape[2] - (gc*2)) {
        i += gc;
        j += shape[1] - gc;
        k += gc;
        p(IDX(i, j, k, shape)) = p(IDX(i, j-1, k, shape));
    }
}

__global__ void kernel_pbc_yminus(const MatrixFrame<REAL> *vp, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &p = *vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - (gc*2) && j < 1 && k < shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        p(IDX(i, j, k, shape)) = p(IDX(i, j+1, k, shape));
    }
}

__global__ void kernel_ubc_xplus(const MatrixFrame<REAL> *vu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - (gc*2) && k < shape[2] - (gc*2)) {
        i += shape[0] - gc;
        j += gc;
        k += gc;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i  , j, k, shape), d) = 2 * uboundary[d] - u(IDX(i-1, j, k, shape), d);
            u(IDX(i+1, j, k, shape), d) = 2 * uboundary[d] - u(IDX(i-2, j, k, shape), d);
        }
    }
}

__global__ void kernel_ubc_xminus(const MatrixFrame<REAL> *vu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - (gc*2) && k < shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i  , j, k, shape), d) = 2 * uboundary[d] - u(IDX(i+1, j, k, shape), d);
            u(IDX(i-1, j, k, shape), d) = 2 * uboundary[d] - u(IDX(i+2, j, k, shape), d);
        }
    }
}

__global__ void kernel_ubc_yplus(const MatrixFrame<REAL> *vu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - (gc*2) && j < 1 && k < shape[2] - (gc*2)) {
        i += gc;
        j += shape[1] - gc;
        k += gc;
        REAL uboundary[] = {1.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i, j  , k, shape), d) = 2 * uboundary[d] - u(IDX(i, j-1, k, shape), d);
            u(IDX(i, j+1, k, shape), d) = 2 * uboundary[d] - u(IDX(i, j-2, k, shape), d);
        }
    }
}

__global__ void kernel_ubc_yminus(const MatrixFrame<REAL> *vu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &u = *vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - (gc*2) && j < 1 && k < shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i, j  , k, shape), d) = 2 * uboundary[d] - u(IDX(i, j+1, k, shape), d);
            u(IDX(i, j-1, k, shape), d) = 2 * uboundary[d] - u(IDX(i, j+2, k, shape), d);
        }
    }
}


__global__ void kernel_uubc_xplus(const MatrixFrame<REAL> *vuu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &uu = *vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - (gc*2) && k < shape[2] - (gc*2)) {
        i += shape[0] - gc - 1;
        j += gc;
        k += gc;
        uu(IDX(i, j, k, shape), 0) = 0;
    }
}

__global__ void kernel_uubc_xminus(const MatrixFrame<REAL> *vuu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &uu = *vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < shape[1] - (gc*2) && k < shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        uu(IDX(i, j, k, shape), 0) = 0;
    }
}

__global__ void kernel_uubc_yplus(const MatrixFrame<REAL> *vuu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &uu = *vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - (gc*2) && j < 1 && k < shape[2] - (gc*2)) {
        i += gc;
        j += shape[1] - gc - 1;
        k += gc;
        uu(IDX(i, j, k, shape), 1) = 0;
    }
}

__global__ void kernel_uubc_yminus(const MatrixFrame<REAL> *vuu, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &uu = *vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] - (gc*2) && j < 1 && k < shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        uu(IDX(i, j, k, shape), 1) = 0;
    }
}

void pbc_xplus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM s) {
    dim3 grid(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_pbc_xplus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_xminus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM s) {
    dim3 grid(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_pbc_xminus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_yplus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_pbc_yplus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void pbc_yminus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_pbc_yminus<<<grid, block, 0, s>>>(p.devptr, pdm.shape, gc);
}

void ubc_xplus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM s) {
    dim3 grid(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_ubc_xplus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void ubc_xminus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM s) {
    dim3 grid(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_ubc_xminus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void ubc_yplus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_ubc_yplus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void ubc_yminus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_ubc_yminus<<<grid, block, 0, s>>>(u.devptr, pdm.shape, gc);
}

void uubc_xplus(Matrix<REAL> &uu, Region &pdm, INT gc, STREAM s) {
    dim3 grid(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_uubc_xplus<<<grid, block, 0, s>>>(uu.devptr, pdm.shape, gc);
}

void uubc_xminus(Matrix<REAL> &uu, Region &pdm, INT gc, STREAM s) {
    dim3 grid(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(1, 8, 8);
    kernel_uubc_xminus<<<grid, block, 0, s>>>(uu.devptr, pdm.shape, gc);
}

void uubc_yplus(Matrix<REAL> &uu, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_uubc_yplus<<<grid, block, 0, s>>>(uu.devptr, pdm.shape, gc);
}

void uubc_yminus(Matrix<REAL> &uu, Region &pdm, INT gc, STREAM s) {
    dim3 grid((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block(8, 1, 8);
    kernel_uubc_yminus<<<grid, block, 0, s>>>(uu.devptr, pdm.shape, gc);
}

__global__ void kernel_copy_z5(const MatrixFrame<REAL> *vv, INT3 shape, INT gc) {
    const MatrixFrame<REAL> &v = *vv;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < shape[0] && j < shape[1] && k < 1) {
        INT idxcc      = IDX(i, j, gc    , shape);
        INT idxzplus1  = IDX(i, j, gc + 1, shape);
        INT idxzplus2  = IDX(i, j, gc + 2, shape);
        INT idxzminus1 = IDX(i, j, gc - 1, shape);
        INT idxzminus2 = IDX(i, j, gc - 2, shape);
        for (INT n = 0; n < v.shape[1]; n ++) {
            v(idxzminus2, n) = v(idxzplus2, n) = v(idxzminus1, n) = v(idxzplus1, n) = v(idxcc, n);
        }
    }
}

void copy_z5(Matrix<REAL> &v, Region &pdm, INT gc) {
    dim3 grid((pdm.shape[0] + 7)/8, (pdm.shape[1] + 7)/8, 1);
    dim3 block(8, 8, 1);
    kernel_copy_z5<<<grid, block>>>(v.devptr, pdm.shape, gc);
}