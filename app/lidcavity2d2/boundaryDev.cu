#include "boundaryDev.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

namespace LidCavity2d2 {

using namespace Falm;

__global__ void kernel_pressureBC_E(
    const MatrixFrame<REAL> *vp,
    INT3              pdm_shape,
    INT                gc
) {
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

__global__ void kernel_pressureBC_W(
    const MatrixFrame<REAL> *vp,
    INT3              pdm_shape,
    INT                gc
) {
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

__global__ void kernel_pressureBC_N(
    const MatrixFrame<REAL> *vp,
    INT3              pdm_shape,
    INT                gc
) {
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

__global__ void kernel_pressureBC_S(
    const MatrixFrame<REAL> *vp,
    INT3              pdm_shape,
    INT                gc
) {
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

void dev_pressureBC_E(
    Matrix<REAL> &p,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_pressureBC_E<<<grid_dim_ew, block_dim_ew, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void dev_pressureBC_W(
    Matrix<REAL> &p,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_pressureBC_W<<<grid_dim_ew, block_dim_ew, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void dev_pressureBC_N(
    Matrix<REAL> &p,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_pressureBC_N<<<grid_dim_ns, block_dim_ns, 0, stream>>>(p.devptr, pdm.shape, gc);
}

void dev_pressureBC_S(
    Matrix<REAL> &p,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_pressureBC_S<<<grid_dim_ns, block_dim_ns, 0, stream>>>(p.devptr, pdm.shape, gc);
}

__global__ void kernel_velocityBC_E(
    const MatrixFrame<REAL> *vu,
    INT3              pdm_shape,
    INT                gc
) {
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

__global__ void kernel_velocityBC_W(
    const MatrixFrame<REAL> *vu,
    INT3              pdm_shape,
    INT                gc
) {
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

__global__ void kernel_velocityBC_N(
    const MatrixFrame<REAL> *vu,
    INT3              pdm_shape,
    INT                gc
) {
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

__global__ void kernel_velocityBC_S(
    const MatrixFrame<REAL> *vu,
    INT3              pdm_shape,
    INT                gc
) {
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

void dev_velocityBC_E(
    Matrix<REAL> &u,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_velocityBC_E<<<grid_dim_ew, block_dim_ew, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void dev_velocityBC_W(
    Matrix<REAL> &u,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_velocityBC_W<<<grid_dim_ew, block_dim_ew, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void dev_velocityBC_N(
    Matrix<REAL> &u,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_velocityBC_N<<<grid_dim_ns, block_dim_ns, 0, stream>>>(u.devptr, pdm.shape, gc);
}

void dev_velocityBC_S(
    Matrix<REAL> &u,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_velocityBC_S<<<grid_dim_ns, block_dim_ns, 0, stream>>>(u.devptr, pdm.shape, gc);
}

__global__ void kernel_forceFaceVelocityZero_E(
    const MatrixFrame<REAL> *vuu,
    INT3              pdm_shape,
    INT                gc
) {
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

__global__ void kernel_forceFaceVelocityZero_W(
    const MatrixFrame<REAL> *vuu,
    INT3              pdm_shape,
    INT                gc
) {
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

__global__ void kernel_forceFaceVelocityZero_N(
    const MatrixFrame<REAL> *vuu,
    INT3              pdm_shape,
    INT                gc
) {
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

__global__ void kernel_forceFaceVelocityZero_S(
    const MatrixFrame<REAL> *vuu,
    INT3              pdm_shape,
    INT                gc
) {
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

void dev_forceFaceVelocityZero_E(
    Matrix<REAL> &uu,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_forceFaceVelocityZero_E<<<grid_dim_ew, block_dim_ew, 0, stream>>>(uu.devptr, pdm.shape, gc);
}

void dev_forceFaceVelocityZero_W(
    Matrix<REAL> &uu,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_forceFaceVelocityZero_W<<<grid_dim_ew, block_dim_ew, 0, stream>>>(uu.devptr, pdm.shape, gc);
}

void dev_forceFaceVelocityZero_N(
    Matrix<REAL> &uu,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_forceFaceVelocityZero_N<<<grid_dim_ns, block_dim_ns, 0, stream>>>(uu.devptr, pdm.shape, gc);
}

void dev_forceFaceVelocityZero_S(
    Matrix<REAL> &uu,
    Region       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_forceFaceVelocityZero_S<<<grid_dim_ns, block_dim_ns, 0, stream>>>(uu.devptr, pdm.shape, gc);
}

}