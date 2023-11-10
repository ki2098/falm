#include "bcdev.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"



namespace LID3D {

using namespace Falm;

__global__ void kernel_ubc_xminus(const MatrixFrame<REAL> *vu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &u=*vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i  , j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i+1, j, k, pdm_shape), d);
            u(IDX(i-1, j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i+2, j, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_ubc_xplus(const MatrixFrame<REAL> *vu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &u=*vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += pdm_shape[0] - gc;
        j += gc;
        k += gc;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i  , j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i-1, j, k, pdm_shape), d);
            u(IDX(i+1, j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i-2, j, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_ubc_yminus(const MatrixFrame<REAL> *vu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &u=*vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i, j  , k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j+1, k, pdm_shape), d);
            u(IDX(i, j-1, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j+2, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_ubc_yplus(const MatrixFrame<REAL> *vu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &u=*vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += pdm_shape[1] - gc;
        k += gc;
        REAL uboundary[] = {1.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i, j  , k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j-1, k, pdm_shape), d);
            u(IDX(i, j+1, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j-2, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_ubc_zminus(const MatrixFrame<REAL> *vu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &u=*vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < pdm_shape[1] - (gc*2) && k < 1) {
        i += gc;
        j += gc;
        k += gc - 1;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i, j, k  , pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j, k+1, pdm_shape), d);
            u(IDX(i, j, k-1, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j, k+2, pdm_shape), d);
        }
    }
}

__global__ void kernel_ubc_zplus(const MatrixFrame<REAL> *vu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &u=*vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < pdm_shape[1] - (gc*2) && k < 1) {
        i += gc;
        j += gc;
        k += pdm_shape[2] - gc;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i, j, k  , pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j, k-1, pdm_shape), d);
            u(IDX(i, j, k+1, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j, k-2, pdm_shape), d);
        }
    }
}

void ubc_xminus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(1, 8, 8);
    kernel_ubc_xminus<<<gdim, bdim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void ubc_xplus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(1, 8, 8);
    kernel_ubc_xplus<<<gdim, bdim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void ubc_yminus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(8, 1, 8);
    kernel_ubc_yminus<<<gdim, bdim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void ubc_yplus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(8, 1, 8);
    kernel_ubc_yplus<<<gdim, bdim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void ubc_zminus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, (pdm.shape[1] - (gc*2) + 7) / 8, 1);
    dim3 bdim(8, 8, 1);
    kernel_ubc_zminus<<<gdim, bdim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void ubc_zplus(Matrix<REAL> &u, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, (pdm.shape[1] - (gc*2) + 7) / 8, 1);
    dim3 bdim(8, 8, 1);
    kernel_ubc_zplus<<<gdim, bdim, 0, stream>>>(u.devptr, pdm.shape, gc);
}

__global__ void kernel_pbc_xminus(MatrixFrame<REAL> *vp, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &p=*vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i+1, j, k, pdm_shape));
    }
}

__global__ void kernel_pbc_xplus(MatrixFrame<REAL> *vp, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &p=*vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += pdm_shape[0] - gc;
        j += gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i-1, j, k, pdm_shape));
    }
}

__global__ void kernel_pbc_yminus(MatrixFrame<REAL> *vp, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &p=*vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j+1, k, pdm_shape));
    }
}

__global__ void kernel_pbc_yplus(MatrixFrame<REAL> *vp, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &p=*vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += pdm_shape[1] - gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j-1, k, pdm_shape));
    }
}

__global__ void kernel_pbc_zminus(MatrixFrame<REAL> *vp, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &p=*vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < pdm_shape[1] - (gc*2) && k < 1) {
        i += gc;
        j += gc;
        k += gc - 1;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j, k+1, pdm_shape));
    }
}

__global__ void kernel_pbc_zplus(MatrixFrame<REAL> *vp, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &p=*vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < pdm_shape[1] - (gc*2) && k < 1) {
        i += gc;
        j += gc;
        k += pdm_shape[2] - gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j, k-1, pdm_shape));
    }
}

void pbc_xminus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(1, 8, 8);
    kernel_pbc_xminus<<<gdim, bdim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void pbc_xplus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(1, 8, 8);
    kernel_pbc_xplus<<<gdim, bdim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void pbc_yminus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(8, 1, 8);
    kernel_pbc_yminus<<<gdim, bdim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void pbc_yplus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(8, 1, 8);
    kernel_pbc_yplus<<<gdim, bdim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void pbc_zminus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, (pdm.shape[1] - (gc*2) + 7) / 8, 1);
    dim3 bdim(8, 8, 1);
    kernel_pbc_zminus<<<gdim, bdim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void pbc_zplus(Matrix<REAL> &p, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, (pdm.shape[1] - (gc*2) + 7) / 8, 1);
    dim3 bdim(8, 8, 1);
    kernel_pbc_zplus<<<gdim, bdim, 0, stream>>>(p.devptr, pdm.shape, gc);
}

__global__ void kernel_uubc_xminus(const MatrixFrame<REAL> *vuu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &uu=*vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 0) = 0;
    }
}

__global__ void kernel_uubc_xplus(const MatrixFrame<REAL> *vuu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &uu=*vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += pdm_shape[0] - gc - 1;
        j += gc;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 0) = 0;
    }
}

__global__ void kernel_uubc_yminus(const MatrixFrame<REAL> *vuu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &uu=*vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 1) = 0;
    }
}

__global__ void kernel_uubc_yplus(const MatrixFrame<REAL> *vuu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &uu=*vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += pdm_shape[1] - gc - 1;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 1) = 0;
    }
}

__global__ void kernel_uubc_zminus(const MatrixFrame<REAL> *vuu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &uu=*vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < pdm_shape[1] - (gc*2) && k < 1) {
        i += gc;
        j += gc;
        k += gc - 1;
        uu(IDX(i, j, k, pdm_shape), 2) = 0;
    }
}

__global__ void kernel_uubc_zplus(const MatrixFrame<REAL> *vuu, INT3 pdm_shape, INT gc) {
    const MatrixFrame<REAL> &uu=*vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < pdm_shape[1] - (gc*2) && k < 1) {
        i += gc;
        j += gc;
        k += pdm_shape[2] - gc - 1;
        uu(IDX(i, j, k, pdm_shape), 2) = 0;
    }
}

void uubc_xminus(Matrix<REAL> &uu, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(1, 8, 8);
    kernel_uubc_xminus<<<gdim, bdim, 0, stream>>>(uu.devptr, pdm.shape, gc);
}

void uubc_xplus(Matrix<REAL> &uu, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(1, 8, 8);
    kernel_uubc_xplus<<<gdim, bdim, 0, stream>>>(uu.devptr, pdm.shape, gc);
}

void uubc_yminus(Matrix<REAL> &uu, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(8, 1, 8);
    kernel_uubc_yminus<<<gdim, bdim, 0, stream>>>(uu.devptr, pdm.shape, gc);
}

void uubc_yplus(Matrix<REAL> &uu, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 bdim(8, 1, 8);
    kernel_uubc_yplus<<<gdim, bdim, 0, stream>>>(uu.devptr, pdm.shape, gc);
}

void uubc_zminus(Matrix<REAL> &uu, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, (pdm.shape[1] - (gc*2) + 7) / 8, 1);
    dim3 bdim(8, 8, 1);
    kernel_uubc_zminus<<<gdim, bdim, 0, stream>>>(uu.devptr, pdm.shape, gc);
}

void uubc_zplus(Matrix<REAL> &uu, Region &pdm, INT gc, STREAM stream) {
    dim3 gdim((pdm.shape[0] - (gc*2) + 7) / 8, (pdm.shape[1] - (gc*2) + 7) / 8, 1);
    dim3 bdim(8, 8, 1);
    kernel_uubc_zplus<<<gdim, bdim, 0, stream>>>(uu.devptr, pdm.shape, gc);
}

}